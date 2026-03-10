from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import WiseConfig


class WISE(Editor):
    def __init__(self, config: WiseConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "right"

    def _apply(self, request: EditRequest):
        raise NotImplementedError("WISE is not implemented yet.")

    def _restore(self):
        raise NotImplementedError("WISE is not implemented yet.")
