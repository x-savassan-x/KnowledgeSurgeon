import torch

from ...config import RRomeConfig
from ..editor import Editor
from ..editor_utils import EditRequest
from .r_rome_main import apply_r_rome_to_model


class RROME(Editor):
    config: RRomeConfig

    def __init__(self, config: RRomeConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "right"
        self._initial_weights: dict[str, torch.Tensor] = {}

    def edit(self, request: EditRequest) -> None:
        ee_request = [{
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.answer,
        }]
        need_reset_copy = not self._initial_weights
        _, weights_copy = apply_r_rome_to_model(
            self.model, self.tokenizer, ee_request, self.config,
            return_orig_weights=need_reset_copy,
        )
        if need_reset_copy:
            self._initial_weights = weights_copy

    def reset(self) -> None:
        if not self._initial_weights:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._initial_weights.items():
                params[name][...] = weight.clone()
        self._initial_weights = {}

    def copy_for_generation(self, target: "RROME") -> None:
        if not isinstance(target, RROME):
            raise TypeError("R-ROME generation copy requires another R-ROME editor.")
        self.copy_named_parameters(self, target, self._initial_weights)
