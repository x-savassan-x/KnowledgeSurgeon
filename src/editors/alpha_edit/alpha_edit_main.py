from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
cache_c_new = False
cache_c_key = None


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    reset_cache=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :param reset_cache: If true, will reset cache_c_new to False, forcing re-initialization of cache_c.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded_path, cache_c, cache_c_new, cache_c_key

    # Reset cache if requested
    if reset_cache:
        cache_c_new = False

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    p_loc = Path(hparams.P_loc)
    if not p_loc.exists():
        raise FileNotFoundError(
            f"AlphaEdit projector is missing: {p_loc}. "
            "Download it with notebook/assets.ipynb before editing."
        )

    resolved_p_loc = p_loc.resolve()
    if P_loaded_path != resolved_p_loc:
        P = torch.load(p_loc, map_location="cpu")
        P_loaded_path = resolved_p_loc

    current_cache_key = (
        hparams.model_key,
        tuple(hparams.layers),
        hparams.rewrite_module_tmp,
    )
    if not cache_c_new or cache_c_key != current_cache_key:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        input_size = W_out.shape[1]
        cache_c = torch.zeros((len(hparams.layers), input_size, input_size), device="cpu")
        del W_out
        cache_c_new = True
        cache_c_key = current_cache_key

    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            w = nethook.get_parameter(model, w_name)
            upd_matrix = to_param_device(upd_m, w)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.to(device=w.device, dtype=w.dtype)

    # print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    cache_template: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

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
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

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
        # Retrieve k/v pair if already stored in cache
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
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(z_device))
                data_loaded = True
            except Exception:
                pass

        # Compute k/v pair if not loaded from cache
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
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                # print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        # print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        layer_device = get_module_device(
            model,
            hparams.rewrite_module_tmp.format(layer),
            fallback=hparams.device,
        )
        # print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
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
        # print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        layer_ks = layer_ks.float()
        targets = targets.float()
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
            P[i, :, :].to(device=layer_device, dtype=torch.float32)
            @ (
                layer_ks.to(device=layer_device, dtype=torch.float32)
                @ layer_ks.T.to(device=layer_device, dtype=torch.float32)
                + cache_c[i, :, :].to(device=layer_device, dtype=torch.float32)
            )
            + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float32, device=layer_device),
            P[i, :, :].to(device=layer_device, dtype=torch.float32)
            @ layer_ks.to(device=layer_device, dtype=torch.float32)
            @ resid.T.to(device=layer_device, dtype=torch.float32)
        )

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        # print("orig norm", torch.linalg.norm(weights[weight_name]))
        # print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            updated_weight = weights[weight_name].to(
                device=upd_matrix.device,
                dtype=upd_matrix.dtype,
            ) + upd_matrix
            copy_to_param(weights[weight_name], updated_weight)
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )

        # Clear GPU memory
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        layer_ks = layer_ks.float()
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            copy_to_param(v, weights_copy[k])

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


def get_context_templates(model, tok):
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
    model,
    tok,
    layer,
    hparams,
    progress=None,
    force_recompute: bool = False,
    status_callback=None,
):
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
