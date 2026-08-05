import torch

from ...config import FtConfig
from ..editor import Editor
from ..editor_utils import EditRequest
from .ft_main import apply_ft_to_model


class FT(Editor):
    config: FtConfig

    def __init__(self, config: FtConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "left"
        self._initial_weights: dict[str, torch.Tensor] = {}

    def edit(self, request: EditRequest) -> None:
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.answer,
        }]
        need_reset_copy = not self._initial_weights
        _, weights_copy = apply_ft_to_model(
            self.model,
            self.tokenizer,
            ee_request,
            self.config,
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

    def copy_for_generation(self, target: "FT") -> None:
        if not isinstance(target, FT):
            raise TypeError("FT generation copy requires another FT editor.")
        self.copy_named_parameters(self, target, self._initial_weights)
