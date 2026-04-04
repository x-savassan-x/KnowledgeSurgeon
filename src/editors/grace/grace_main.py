from copy import deepcopy
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from .grace_core import GraceCore
from .utils import tokenize


def apply_grace_to_model(
    model,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[GraceCore, Any]:
    request = requests[0]
    if copy:
        model = deepcopy(model)

    editor = GraceCore(model=model, config=hparams, device=hparams.device)
    tokens = tokenize(request, tokenizer=tok, device=hparams.device)
    editor.edit(config=hparams, tokens=tokens, edit_id=request["target_new"])

    weights_copy = editor.reset_layer
    return editor, weights_copy
