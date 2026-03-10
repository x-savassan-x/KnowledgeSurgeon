from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.nethook import get_parameter
from ...config import FtConfig


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config: FtConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute weight deltas via fine-tuning, apply them to the model.
    Returns dict mapping weight names → original weight tensors (for restore).
    """
    deltas = execute_ft(model, tok, requests, config)

    weights_copy = {}
    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = get_parameter(model, w_name)
            weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config: FtConfig,
) -> Dict[str, torch.Tensor]:
    """
    Fine-tune specified layers on the given requests.
    Returns weight deltas (model is restored to original state).
    """
    device = next(model.parameters()).device

    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Collect the parameters we want to update
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in config.layers
        if config.rewrite_module_tmp.format(layer) in n
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    opt = torch.optim.Adam(
        list(weights.values()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    loss_meter = _AverageMeter()
    for it in range(config.num_steps):
        print(20 * "=" + f" Epoch: {it} " + 20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            _chunks(texts, config.batch_size),
            _chunks(targets, config.batch_size),
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)[
                "input_ids"
            ].to(device)

            if config.objective_optimization == "prompt_last":
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                if tok.unk_token_id is not None:
                    loss_mask = torch.ne(target_ids, tok.unk_token_id)
                else:
                    loss_mask = torch.ones_like(target_ids, dtype=torch.bool)
            elif config.objective_optimization == "target_new":
                inputs_targets = [t + g for t, g in zip(txt, tgt)]
                inputs_targets = tok(
                    inputs_targets, return_tensors="pt", padding=True
                ).to(device)
                num_prompt_toks = [
                    int((i != tok.pad_token_id).sum())
                    for i in inputs["input_ids"].cpu()
                ]
                num_pad_toks = [
                    int((i == tok.pad_token_id).sum())
                    for i in inputs_targets["input_ids"].cpu()
                ]
                prompt_len = [
                    x + y for x, y in zip(num_pad_toks, num_prompt_toks)
                ]
                prompt_target_len = inputs_targets["input_ids"].size(1)
                label_mask = torch.tensor(
                    [
                        [False] * length + [True] * (prompt_target_len - length)
                        for length in prompt_len
                    ]
                ).to(device)
            else:
                raise NotImplementedError(
                    f"{config.objective_optimization} not supported"
                )

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]

            if config.objective_optimization == "prompt_last":
                probs = torch.nn.functional.log_softmax(
                    model(**inputs).logits[torch.arange(bs), last_token_inds],
                    dim=-1,
                )
                loss = -(
                    torch.gather(probs, 1, target_ids) * loss_mask
                ).sum(1) / loss_mask.sum(1)
                loss = loss.mean()
            elif config.objective_optimization == "target_new":
                logits = model(**inputs_targets).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs_targets["input_ids"][..., 1:].contiguous()
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss.view(bs, -1)
                loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                loss = loss.mean()

            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(config.norm_constraint) is float:
                eps = config.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")
        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore original weights
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas


def _chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class _AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
