import copy

import torch

from .wise_main import apply_wise_to_model
from .wise_adapter import WISEModule, WISEAdapter

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import WiseConfig


class WISE(Editor):
    config: WiseConfig

    def __init__(self, config: WiseConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "left"
        self.config.device = self.config.gpu_id
        self.config.sequential_edit = True
        self._wise_editor: WISEModule | None = None
        self._initial_adapter_state: dict | None = None

    def reset(self):
        if self._wise_editor is not None:
            self._wise_editor.reset_layer()
            self._wise_editor = None
            self._initial_adapter_state = None

    def _apply(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": " " + request.new_ans,
            "loc_prompt": request.subject,
        }]

        self._wise_editor, _ = apply_wise_to_model(
            self.model, self.tokenizer, ee_request, self.config,
        )

    def _update_checkpoint(self):
        pass

    def _revert(self):
        pass
