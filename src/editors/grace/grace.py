import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable

from .grace_core import GRACECore
from .grace_utils import tokenize
from ..editor import Editor
from ...hparams import HyperParams, GraceHyperParams
from ...utils import EditRequest


class GRACE(Editor):
    def __init__(self, hparams: HyperParams):
        super().__init__(hparams)
        self.tokenizer.padding_side = 'left'

    def _apply(self, edit_request: EditRequest):
        edited_model, weights_copy = apply_grace_to_model(
            self.model,
            self.tokenizer,
            edit_request,
            self.hparams,
        )
        self.model = edited_model
        self._weights_copy = weights_copy

    def _restore(self):
        if self._weights_copy is not None:
            self._weights_copy()
            self._weights_copy = None


def apply_grace_to_model(model: AutoModelForCausalLM,
                         tok: AutoTokenizer,
                         request: EditRequest,
                         hparams: GraceHyperParams) -> tuple[GRACECore, Callable]:
    device = torch.device(f"cuda:{hparams.gpu_id}")
    editor = GRACECore(model=model, config=hparams, device=device)
    tokens = tokenize(request, tokenizer=tok, device=device)
    editor.edit(config=hparams, tokens=tokens, edit_id=request.new_ans)

    return editor, editor.reset_layer
