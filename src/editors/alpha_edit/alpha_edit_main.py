from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.devices import copy_to_param, get_module_device, to_param_device
from ...utils.generate import generate_fast
from ...utils import nethook
from ...utils.layer_stats import layer_stats
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P = None
P_loaded_path: Path | None = None
cache_c: torch.Tensor | None = None
cache_c_new = False
cache_c_key = None


def load_AlphaEdit_projector(hparams) -> torch.Tensor:
    """Load the configured projector once and return its CPU tensor."""
    global P, P_loaded_path

    projector_path = Path(hparams.P_loc)
    if not projector_path.exists():
        raise FileNotFoundError(
            f"AlphaEdit projector is missing: {projector_path}. "
            "Download it with notebook/assets.ipynb before editing."
        )
    resolved_path = projector_path.resolve()
    if P is None or P_loaded_path != resolved_path:
        P = torch.load(projector_path, map_location="cpu")
        P_loaded_path = resolved_path
    return P


def _alpha_edit_cache_key(hparams) -> tuple:
    return (
        hparams.model_key,
        tuple(hparams.layers),
        hparams.rewrite_module_tmp,
    )


def _resolve_cache_matrix(
    model: AutoModelForCausalLM,
    hparams,
) -> torch.Tensor:
    global cache_c, cache_c_new, cache_c_key

    current_key = _alpha_edit_cache_key(hparams)
    if cache_c is None or not cache_c_new or cache_c_key != current_key:
        output_weight = nethook.get_parameter(
            model,
            f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight",
        )
        input_size = output_weight.shape[1]
        cache_c = torch.zeros(
            (len(hparams.layers), input_size, input_size),
            device="cpu",
        )
        cache_c_new = True
        cache_c_key = current_key
    return cache_c


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    keep_original_weight: bool = False,
    reset_cache: bool = False,
    **kwargs,
) -> tuple[AutoModelForCausalLM, dict[str, torch.Tensor]]:
    """Apply an AlphaEdit update and return optional reset weights."""

    global cache_c, cache_c_key, cache_c_new

    if reset_cache:
        cache_c = None
        cache_c_new = False
        cache_c_key = None

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    load_AlphaEdit_projector(hparams)

    cache_matrix = _resolve_cache_matrix(model, hparams)
    deltas = execute_AlphaEdit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        cache_matrix=cache_matrix,
    )

    with torch.no_grad():
        for weight_name, update in deltas.items():
            weight = nethook.get_parameter(model, weight_name)
            update = upd_matrix_match_shape(update, weight.shape)
            if return_orig_weights:
                weights_copy[weight_name] = weight.detach().clone()
            weight.add_(to_param_device(update, weight, dtype=weight.dtype))

    # print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    cache_template: Optional[str] = None,
    cache_matrix: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    if cache_matrix is None:
        cache_matrix = _resolve_cache_matrix(model, hparams)

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'].lower() in request['prompt'].lower(), f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}"
            if request['subject'] in request['prompt']:
                requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
            else:
                requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'].capitalize(), '{}')
        # print(
        #     f"Executing AlphaEdit algo for: "
        #     f"[{request['prompt']}] -> [{request['target_new']}]"
        # )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {
        name: weight.detach().clone() for name, weight in weights.items()
    }

    try:
        # Compute z for final layer
        context_templates = get_context_templates(model, tok)
        z_layer = hparams.layers[-1]
        z_device = get_module_device(
            model,
            hparams.layer_module_tmp.format(z_layer),
            fallback=hparams.device,
        )
        z_list = []

        for request in requests:
            # Retrieve k/v pair if already stored in cache.
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if cache_fname is not None and cache_fname.exists():
                try:
                    data = np.load(cache_fname)
                    z_list.append(torch.from_numpy(data["v_star"]).to(z_device))
                    data_loaded = True
                except Exception:
                    pass

            # Compute k/v pair if not loaded from cache.
            if not data_loaded:
                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                z_list.append(cur_z)

                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{"v_star": cur_z.detach().cpu().numpy()},
                    )
        zs = torch.stack(z_list, dim=1)

        # Insert
        for i, layer in enumerate(hparams.layers):
            # Get current model activations.
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            layer_device = get_module_device(
                model,
                hparams.rewrite_module_tmp.format(layer),
                fallback=hparams.device,
            )
            # Compute residual error.
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs.to(cur_zs.device) - cur_zs
            repeat_factor = layer_ks.size(1) // targets.size(1)
            targets = targets.repeat_interleave(repeat_factor, dim=1).float()
            layer_ks = layer_ks.float()
            resid = targets / (len(hparams.layers) - i)
            projector = P[i].to(device=layer_device, dtype=torch.float32)
            layer_keys = layer_ks.to(device=layer_device, dtype=torch.float32)
            system = projector @ (
                layer_keys @ layer_keys.T
                + cache_matrix[i].to(device=layer_device, dtype=torch.float32)
            )
            system += hparams.L2 * torch.eye(
                layer_ks.shape[0],
                dtype=torch.float32,
                device=layer_device,
            )
            rhs = projector @ layer_keys @ resid.T.to(
                device=layer_device,
                dtype=torch.float32,
            )
            upd_matrix = torch.linalg.solve(system, rhs)

            # Adjust update matrix shape.
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Apply the layer update temporarily so later layers see the candidate.
            with torch.no_grad():
                deltas[weight_name] = upd_matrix.detach().cpu()
                weights[weight_name].add_(upd_matrix)

            # Release solve intermediates before processing the next layer.
            del layer_ks, layer_keys, cur_zs, targets, resid
            del projector, system, rhs, upd_matrix
            torch.cuda.empty_cache()

        for i, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(
                model,
                tok,
                requests,
                hparams,
                layer,
                context_templates,
            ).T.float().cpu()
            cache_matrix[i].addmm_(layer_ks, layer_ks.T)
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                copy_to_param(weight, weights_copy[name])

    # print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
    progress=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    key = (
        hparams.model_key,
        layer_name,
        str(hparams.stats_dir),
        mom2_dataset,
        int(mom2_n_samples),
        mom2_dtype,
        hparams.layer_stats_max_length,
    )

    # print(f"Retrieving covariance statistics for {hparams.model_key} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
            progress=progress,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(get_module_device(model, layer_name, fallback=hparams.device)))
        if inv
        else COV_CACHE[key].to(get_module_device(model, layer_name, fallback=hparams.device))
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Orient an update matrix to match the edited linear weight."""
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
) -> List[List[str]]:
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        # print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def get_project(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    hparams,
    progress=None,
    force_recompute: bool = False,
    status_callback=None,
) -> torch.Tensor:
    if status_callback is not None:
        status_callback("covariance")
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams,
        progress=progress,
    ).cpu()
    if status_callback is not None:
        status_callback("svd")
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    if status_callback is not None:
        status_callback("project")
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print(len(small_singular_indices))
    result = U[:, small_singular_indices] @ U[:, small_singular_indices].T
    if status_callback is not None:
        status_callback("done")
    return result
