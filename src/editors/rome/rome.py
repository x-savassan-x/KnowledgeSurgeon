import torch

from ..editor import Editor
from ...config import Config
from ...utils import EditRequest
from ...utils.nethook import get_parameter

from .rome_core import apply_rome_to_model


class ROME(Editor):
    def __init__(self, config: Config):
        super().__init__(config)
        self.tokenizer.padding_side = 'right'

    def _apply(self, edit_request: EditRequest):
        request = {
            "prompt": edit_request.prompt,
            "subject": edit_request.subject,
            "target_new": edit_request.new_ans,
        }
        self._weights_copy = apply_rome_to_model(
            self.model, self.tokenizer, request, self.config
        )

    def _restore(self):
        if self._weights_copy is not None:
            with torch.no_grad():
                for w_name, original_w in self._weights_copy.items():
                    w = get_parameter(self.model, w_name)
                    w[...] = original_w
            self._weights_copy = None
