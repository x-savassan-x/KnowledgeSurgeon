from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import AlphaEditConfig


class AlphaEdit(Editor):
    def __init__(self, config: AlphaEditConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "right"

    def _apply(self, request: EditRequest):
        raise NotImplementedError("AlphaEdit is not implemented yet.")

    def _restore(self):
        raise NotImplementedError("AlphaEdit is not implemented yet.")
