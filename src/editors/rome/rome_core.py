from copy import deepcopy
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils import nethook
from ...utils.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from ...config.rome import RomeConfig

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
) -> Dict[str, torch.Tensor]:
    """
    Apply a single ROME edit.  Returns dict mapping weight names → original
    weight tensors (for later restoration).
    """
    weights_copy = {}
    deltas = execute_rome(model, tok, request, config)

    with torch.no_grad():
        for w_name, (delta_u, delta_v) in deltas.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    config: RomeConfig,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes rank-1 deltas for each layer. Model is restored to its
    original state before returning.
    """
    request = deepcopy(request)
    if request["target_new"] != " ":
        request["target_new"] = " " + request["target_new"]

    if "{}" not in request["prompt"]:
        assert request["subject"] in request["prompt"], (
            f"Subject:{request['subject']} not in prompt: {request['prompt']}"
        )
        request["prompt"] = request["prompt"].replace(request["subject"], "{}")

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> "
        f"[{request['target_new']}]"
    )

    # Retrieve and save weights
    weights = {
        f"{config.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{config.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in config.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute deltas layer by layer
    deltas = {}
    for layer in sorted(config.layers):
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            config,
            layer,
            get_context_templates(model, tok, config.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)

        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            config,
            layer,
            left_vector,
            get_context_templates(model, tok, config.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            weight_name = f"{config.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore original weights
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError.
    """
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
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
