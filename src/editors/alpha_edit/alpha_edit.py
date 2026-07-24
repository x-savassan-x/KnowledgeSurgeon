import torch

from ...config import AlphaEditConfig
from ..editor import Editor
from ..editor_utils import EditRequest
from . import alpha_edit_main as _ae_mod
from .alpha_edit_main import apply_AlphaEdit_to_model


class AlphaEdit(Editor):
    def __init__(self, config: AlphaEditConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "right"
        self._initial_weights: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        if not self._initial_weights:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._initial_weights.items():
                params[name][...] = weight.clone()
        # Reset AlphaEdit's cumulative cache so it reinitializes on next edit
        _ae_mod.cache_c_new = False

    def edit(self, request: EditRequest) -> None:
        ee_request = [{
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
        }]
        _, weights_copy = apply_AlphaEdit_to_model(
            self.model, self.tokenizer, ee_request, self.config,
            return_orig_weights=True,
        )
        if not self._initial_weights:
            self._initial_weights = {k: v.detach().clone() for k, v in weights_copy.items()}
