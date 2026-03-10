from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import MemitConfig


class MEMIT(Editor):
    def __init__(self, config: MemitConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "right"

    def _apply(self, request: EditRequest):
        raise NotImplementedError("MEMIT is not implemented yet.")

    def _restore(self):
        raise NotImplementedError("MEMIT is not implemented yet.")
