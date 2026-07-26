from copy import deepcopy
from typing import Any, Iterator

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils import nethook
from ...utils.devices import get_model_device


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict[str, str]],
    hparams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> tuple[AutoModelForCausalLM, dict[str, Any]]:
    """Apply an FT update and return copies of the weights that changed."""
    weights_copy: dict[str, torch.Tensor] = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for weight_name, update in deltas.items():
            weight = nethook.get_parameter(model, weight_name)
            if return_orig_weights and weight_name not in weights_copy:
                weights_copy[weight_name] = weight.detach().clone()
            weight[...] += update

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict[str, str]],
    hparams,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Compute a causal-LM fine-tuning update without retaining it in the model."""
    device = get_model_device(model, fallback=hparams.device)
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            request["target_new"] = " " + request["target_new"]

    weights = {
        name: parameter
        for name, parameter in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in name
    }
    if not weights:
        raise ValueError("FT did not match any parameters to edit.")

    weights_copy = {name: value.detach().clone() for name, value in weights.items()}
    texts = [request["prompt"] for request in requests]
    targets = [request["target_new"] for request in requests]

    optimizer = torch.optim.Adam(
        weights.values(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, weight in model.named_parameters():
        weight.requires_grad = name in weights

    loss_meter = AverageMeter()
    for _ in range(hparams.num_steps):
        loss_meter.reset()

        for text_batch, target_batch in zip(
            chunks(texts, hparams.batch_size),
            chunks(targets, hparams.batch_size),
        ):
            optimizer.zero_grad()
            prompt_tokens = tok(
                text_batch,
                return_tensors="pt",
                padding=True,
            ).to(device)
            batch_count = prompt_tokens["input_ids"].shape[0]

            if hparams.objective_optimization == "prompt_last":
                target_ids = tok(
                    target_batch,
                    return_tensors="pt",
                    padding=True,
                )["input_ids"].to(device)
                token_positions = torch.arange(
                    prompt_tokens["input_ids"].shape[1],
                    device=device,
                )
                last_token_indices = (
                    prompt_tokens["attention_mask"] * token_positions
                ).argmax(dim=1)
                if tok.unk_token_id is not None:
                    target_mask = target_ids != tok.unk_token_id
                else:
                    target_mask = torch.ones_like(target_ids, dtype=torch.bool)
                if tok.pad_token_id is not None:
                    target_mask &= target_ids != tok.pad_token_id

                log_probs = torch.nn.functional.log_softmax(
                    model(**prompt_tokens).logits[
                        torch.arange(batch_count, device=device),
                        last_token_indices,
                    ],
                    dim=-1,
                )
                loss = -(
                    torch.gather(log_probs, 1, target_ids) * target_mask
                ).sum(1) / target_mask.sum(1).clamp_min(1)
                loss = loss.mean()
            elif hparams.objective_optimization == "target_new":
                full_tokens = tok(
                    [
                        text + target
                        for text, target in zip(text_batch, target_batch)
                    ],
                    return_tensors="pt",
                    padding=True,
                ).to(device)
                prompt_token_counts = [
                    int((token_ids != tok.pad_token_id).sum())
                    for token_ids in prompt_tokens["input_ids"].cpu()
                ]
                full_padding_counts = [
                    int((token_ids == tok.pad_token_id).sum())
                    for token_ids in full_tokens["input_ids"].cpu()
                ]
                prompt_lengths = [
                    padding + prompt
                    for padding, prompt in zip(
                        full_padding_counts,
                        prompt_token_counts,
                    )
                ]
                sequence_length = full_tokens["input_ids"].size(1)
                label_mask = torch.tensor(
                    [
                        [False] * length
                        + [True] * (sequence_length - length)
                        for length in prompt_lengths
                    ],
                    device=device,
                )

                logits = model(**full_tokens).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = full_tokens["input_ids"][..., 1:].contiguous()
                token_loss = CrossEntropyLoss(reduction="none")(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(batch_count, -1)
                target_mask = label_mask[:, 1:]
                loss = (
                    (token_loss * target_mask).sum(1)
                    / target_mask.sum(1).clamp_min(1)
                ).mean()
            else:
                raise ValueError(
                    "Unsupported FT objective_optimization: "
                    f"{hparams.objective_optimization!r}."
                )
            loss_meter.update(loss.item(), n=batch_count)

            if loss.item() >= 1e-2:
                loss.backward()
                optimizer.step()

            if isinstance(hparams.norm_constraint, float):
                epsilon = hparams.norm_constraint
                with torch.no_grad():
                    for name, value in weights.items():
                        value[...] = torch.clamp(
                            value,
                            min=weights_copy[name] - epsilon,
                            max=weights_copy[name] + epsilon,
                        )

        if loss_meter.avg < 1e-2:
            break

    deltas = {
        name: (weights[name] - original).detach()
        for name, original in weights_copy.items()
    }
    with torch.no_grad():
        for name, value in weights.items():
            value[...] = weights_copy[name]

    return deltas


def chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive chunks of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
