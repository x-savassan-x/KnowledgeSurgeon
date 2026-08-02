from copy import deepcopy
from typing import Any

from transformers import AutoTokenizer

from ...utils.devices import get_model_device
from .grace_core import GraceCore
from .utils import tokenize


def apply_grace_to_model(
    model,
    tok: AutoTokenizer,
    requests: list[dict],
    hparams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs,
) -> tuple[GraceCore, Any]:
    request = requests[0]
    if copy:
        model = deepcopy(model)

    input_device = get_model_device(model, fallback=hparams.device)
    hparams.device = input_device
    editor = GraceCore(model=model, config=hparams, device=input_device)
    tokens = tokenize(request, tokenizer=tok, device=input_device)
    editor.edit(config=hparams, tokens=tokens, edit_id=request["target_new"])

    weights_copy = editor.reset_layer
    return editor, weights_copy
