from .grace_main import apply_grace_to_model
from .grace_core import GraceCore
from .utils import tokenize

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import GraceConfig


class GRACE(Editor):
    config: GraceConfig

    def __init__(self, config: GraceConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "left"
        self._base_model = self.model
        self._grace_model: GraceCore | None = None
        self._reset_layer = None

    def edit(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
            "subject": request.subject,
        }]

        if self._grace_model is None:
            grace_model, reset_layer = apply_grace_to_model(
                self._base_model,
                self.tokenizer,
                ee_request,
                self.config,
                return_orig_weights=True,
            )
            self._grace_model = grace_model
            self.model = grace_model
            self._reset_layer = reset_layer
        else:
            tokens = tokenize(ee_request[0], tokenizer=self.tokenizer, device=self.device)
            self._grace_model.edit(config=self.config, tokens=tokens, edit_id=ee_request[0]["target_new"])

        self._grace_model.set_inference_mode()

    def reset(self):
        if self._reset_layer is not None:
            self._reset_layer()
            self._reset_layer = None
        self._grace_model = None
        self.model = self._base_model
