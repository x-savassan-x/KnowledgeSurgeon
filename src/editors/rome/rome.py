import torch

from .rome_main import apply_rome_to_model

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import RomeConfig


class ROME(Editor):
    def __init__(self, config: RomeConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "right"
        self.config.device = self.config.gpu_id  # bridge for EasyEdit
        self._initial_weights: dict[str, torch.Tensor] = {}

    def reset(self):
        if not self._initial_weights:
            return
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._initial_weights.items():
                params[name][...] = weight.clone()

    def _apply(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt.replace(request.subject, "{}"),
            "subject": request.subject,
            "target_new": request.new_ans,
            "target_true": request.old_ans,
        }]
        _, weights_copy = apply_rome_to_model(
            self.model, self.tokenizer, ee_request, self.config,
            return_orig_weights=True,
        )
        if not self._initial_weights:
            self._initial_weights = {k: v.detach().clone() for k, v in weights_copy.items()}

    def _update_checkpoint(self):
        pass

    def _revert(self):
        pass

