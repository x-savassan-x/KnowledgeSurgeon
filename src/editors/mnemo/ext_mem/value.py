import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....config.mnemo.mnemo_config import MnemoDeltaConfig
from ....utils import nethook
from ...editor_utils import EditRequest
from .interceptor import Interceptor


def request_token_ids(
    tokenizer: AutoTokenizer,
    request: EditRequest,
    config: MnemoDeltaConfig,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(request.prompt, add_special_tokens=True)
    target = request.answer
    if not target:
        raise ValueError("Mnemo edit targets must not be empty.")
    if (
        config.target_leading_space
        and request.prompt
        and not request.prompt[-1].isspace()
        and not target[0].isspace()
    ):
        target = f" {target}"
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not prompt_ids or not target_ids:
        raise ValueError("Mnemo prompt and target must produce at least one token.")
    return list(prompt_ids), list(target_ids)


class InterventionTrainer:
    """Learns one decoder-block intervention value for an edit request."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        interceptor: Interceptor,
        config: MnemoDeltaConfig,
        *,
        module_name: str,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.interceptor = interceptor
        self.config = config
        self.module_name = module_name
        self.device = device

    def train(self, request: EditRequest) -> tuple[torch.Tensor, float, float]:
        prompt_ids, target_ids = request_token_ids(
            self.tokenizer,
            request,
            self.config,
        )
        input_ids = torch.tensor(
            [prompt_ids + target_ids],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[:, len(prompt_ids):] = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=self.device,
        )

        baseline = self._intervention_state(
            input_ids,
            attention_mask,
            len(prompt_ids) - 1,
        ).float()
        delta = torch.nn.Parameter(
            torch.zeros(
                (1, self._hidden_size()),
                dtype=torch.float32,
                device=self.device,
            )
        )
        optimizer = torch.optim.Adam(
            [delta],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            foreach=False,
        )
        was_training = self.model.training
        self.model.eval()
        final_loss = float("inf")
        try:
            with self.interceptor.arm_training_delta(delta, len(prompt_ids) - 1):
                for _ in range(self.config.num_steps):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = outputs.loss.float()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Mnemo delta loss became non-finite.")
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        norm = torch.linalg.vector_norm(delta)
                        if norm > self.config.max_norm:
                            delta.mul_(self.config.max_norm / norm)
                    final_loss = float(loss.detach().cpu())
                    del outputs, loss
                    if final_loss <= self.config.early_stop_loss:
                        break
        finally:
            self.model.train(was_training)
            del optimizer, input_ids, attention_mask, labels

        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Mnemo delta became non-finite.")
        delta_norm = float(torch.linalg.vector_norm(delta.detach().float()))
        block_dtype = next(nethook.get_module(self.model, self.module_name).parameters()).dtype
        value = baseline + delta.detach()
        return (
            value.to(device="cpu", dtype=block_dtype),
            delta_norm,
            final_loss,
        )

    def _intervention_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        block = nethook.get_module(self.model, self.module_name)
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output) -> None:
            hidden = nethook.get_hidden_state(output)
            captured.append(hidden[:, token_index, :].detach())

        handle = block.register_forward_hook(capture)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
            self.model.train(was_training)
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one intervention state, captured {len(captured)}."
            )
        return captured[0]

    def _hidden_size(self) -> int:
        model_config = getattr(self.model.config, "text_config", None)
        model_config = model_config or self.model.config
        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("The language model config has no hidden_size.")
        return int(hidden_size)
