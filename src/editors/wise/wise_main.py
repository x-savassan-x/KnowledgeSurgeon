from copy import deepcopy
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.devices import get_model_device
from .wise_adapter import WISEModule
from .wise_utils import tokenize, get_context_templates


def apply_wise_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy=False,
    **kwargs: Any,
) -> Tuple[WISEModule, dict]:
    if copy:
        model = deepcopy(model)

    device = get_model_device(model, fallback=hparams.device)
    hparams.device = device
    context_templates = get_context_templates(
        model, tok,
        length_params=[[5, 5], [10, 5]],
        device=device,
    )
    editor = WISEModule(model=model, config=hparams, device=device)

    tokens, act_mask, deact_mask = tokenize(
        requests, tokenizer=tok, device=device,
        context_templates=context_templates, hparams=hparams,
    )
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer
    return editor, weights_copy
