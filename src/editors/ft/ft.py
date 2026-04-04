import torch

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import FtConfig
from .ft_main import apply_ft_to_model


class FT(Editor):
    config: FtConfig

    def __init__(self, config: FtConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "left"
        self._initial_weights: dict[str, torch.Tensor] = {}

    def edit(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
        }]
        _, weights_copy = apply_ft_to_model(
            self.model,
            self.tokenizer,
            ee_request,
            self.config,
            return_orig_weights=True,
        )
        if not self._initial_weights:
            self._initial_weights = {
                k: v.detach().clone() for k, v in weights_copy.items()
            }

    def reset(self):
        if not self._initial_weights:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._initial_weights.items():
                params[name][...] = weight.clone()
