from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.devices import to_param_device
from ...utils.generate import generate_fast
from ...utils import nethook
from .compute_u import compute_u
from .compute_v import compute_v

CONTEXT_TEMPLATES_CACHE = None


def apply_r_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    if not hparams.original_implementation and (
        hparams.enable_prompt_keys == hparams.enable_random_prefix_keys
    ):
        raise ValueError(
            "R-ROME requires exactly one of enable_prompt_keys or "
            "enable_random_prefix_keys unless original_implementation is enabled."
        )

    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    deltas = execute_r_rome(model, tok, request, hparams)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            w = nethook.get_parameter(model, w_name)
            delta_u = to_param_device(delta_u, w)
            delta_v = to_param_device(delta_v, w)
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        # print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_r_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    request = deepcopy(request)
    if request["target_new"] != " ":
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'].lower() in request['prompt'].lower(), f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}"
        if request['subject'] in request['prompt']:
            request['prompt'] = request['prompt'].replace(request['subject'], '{}')
        else:
            request['prompt'] = request['prompt'].replace(request['subject'].capitalize(), '{}')

    # print(
    #     f"Executing ROME algorithm for the update: "
    #     f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    # )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    deltas = {}
    for layer in sorted(hparams.layers):
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        # print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        # print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            left_vector = to_param_device(left_vector, weights[weight_name])
            right_vector = to_param_device(right_vector, weights[weight_name])
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    # print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        # print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
