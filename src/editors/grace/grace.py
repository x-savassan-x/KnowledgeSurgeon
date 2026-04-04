from .grace_main import apply_grace_to_model

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import GraceConfig


class GRACE(Editor):
    config: GraceConfig

    def __init__(self, config: GraceConfig):
        super().__init__(config)
        # EasyEdit defaults GRACE to left padding for autoregressive tokenization.
        self.tokenizer.padding_side = "left"
        self._reset_layer = None

    def edit(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
            "subject": request.subject,
        }]
        _, reset_layer = apply_grace_to_model(
            self.model,
            self.tokenizer,
            ee_request,
            self.config,
            return_orig_weights=True,
        )
        # Capture the restore handle from the first adapter injection.
        if self._reset_layer is None:
            self._reset_layer = reset_layer

    def reset(self):
        if self._reset_layer is not None:
            self._reset_layer()
            self._reset_layer = None
