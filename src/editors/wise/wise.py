from .wise_main import apply_wise_to_model
from .wise_adapter import WISEModule

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import WiseConfig


class WISE(Editor):
    config: WiseConfig

    def __init__(self, config: WiseConfig, model=None, tokenizer=None):
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "left"
        self.config.sequential_edit = True
        self._wise_editor: WISEModule | None = None

    def reset(self):
        if self._wise_editor is not None:
            self._wise_editor.reset_layer()
            self._wise_editor = None
 
    def edit(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "loc_prompt": request.subject,
        }]

        self._wise_editor, _ = apply_wise_to_model(
            self.model, self.tokenizer, ee_request, self.config,
        )
