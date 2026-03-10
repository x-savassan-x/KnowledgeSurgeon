from .wrapper import Wrapper
from .utils import tokenize
from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import GraceConfig


class GRACE(Editor):
    def __init__(self, config: GraceConfig):
        super().__init__(config)
        self.tokenizer.padding_side = 'left'

    def _apply(self, request: EditRequest):
        device = self.model.device
        tokens = tokenize(request, tokenizer=self.tokenizer, device=device)

        if not isinstance(self.model, Wrapper):
            self.model = Wrapper(self.model, self.config)

        self.model.edit(config=self.config, tokens=tokens, edit_id=request.new_ans)
        self._weights_copy = self.model.reset_layer

    def _restore(self):
        if self._weights_copy is not None:
            self._weights_copy()
            self._weights_copy = None
