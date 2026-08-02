from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....utils.devices import copy_to_param, get_module_device, to_param_device
from ....utils.generate import generate_fast
from ....utils import nethook
from ....utils.layer_stats import layer_stats
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


@dataclass
class AlphaEditCacheState:
    matrix: torch.Tensor | None = None
    key: tuple | None = None

    def reset(self) -> None:
        self.matrix = None
        self.key = None


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


def release_AlphaEdit_projector(projector_path: str | Path | None = None) -> bool:
    """Release the process-wide projector when it matches the requested asset."""
    global P, P_loaded_path

    if projector_path is not None and P_loaded_path is not None:
        if Path(projector_path).resolve() != P_loaded_path:
            return False
    P = None
    P_loaded_path = None
    return True


def _alpha_edit_cache_key(hparams) -> tuple:
    return (
        hparams.model_key,
        tuple(hparams.layers),
        hparams.rewrite_module_tmp,
    )


def _resolve_cache_matrix(
    model: AutoModelForCausalLM,
    hparams,
    cache_state: AlphaEditCacheState | None,
) -> torch.Tensor:
    global cache_c, cache_c_new, cache_c_key

    current_key = _alpha_edit_cache_key(hparams)
    if cache_state is not None:
        if cache_state.matrix is None or cache_state.key != current_key:
            output_weight = nethook.get_parameter(
                model,
                f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight",
            )
            input_size = output_weight.shape[1]
            cache_state.matrix = torch.zeros(
                (len(hparams.layers), input_size, input_size),
                device="cpu",
            )
            cache_state.key = current_key
        return cache_state.matrix

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


def _validate_cache_factors(
    cache_matrix: torch.Tensor,
    factors: list[torch.Tensor],
) -> list[torch.Tensor]:
    if cache_matrix.ndim != 3 or cache_matrix.shape[1] != cache_matrix.shape[2]:
        raise ValueError("AlphaEdit cache must contain square matrices per layer.")
    if len(factors) != cache_matrix.shape[0]:
        raise ValueError("AlphaEdit cache factor count does not match configured layers.")

    input_size = cache_matrix.shape[1]
    prepared_factors: list[torch.Tensor] = []
    for layer_index, factor in enumerate(factors):
        if factor.ndim != 2 or factor.shape[0] != input_size:
            raise ValueError(
                "AlphaEdit cache factor has the wrong shape at layer index "
                f"{layer_index}: expected ({input_size}, n), got {tuple(factor.shape)}."
            )
        prepared = factor.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(prepared).all():
            raise ValueError(
                f"AlphaEdit cache factor contains non-finite values at layer index {layer_index}."
            )
        prepared_factors.append(prepared)
    return prepared_factors


def _commit_cache_factors(
    cache_matrix: torch.Tensor,
    factors: list[torch.Tensor],
) -> None:
    prepared_factors = _validate_cache_factors(cache_matrix, factors)
    for layer_index, factor in enumerate(prepared_factors):
        cache_matrix[layer_index].addmm_(factor, factor.T)


def commit_AlphaEdit_cache_factors(
    cache_state: AlphaEditCacheState,
    factors: list[torch.Tensor],
) -> None:
    if cache_state.matrix is None:
        raise RuntimeError("AlphaEdit cache is not initialized.")
    _commit_cache_factors(cache_state.matrix, factors)


def restore_AlphaEdit_weights(
    model: AutoModelForCausalLM,
    weights: dict[str, torch.Tensor],
) -> None:
    """Restore only the parameters changed by an AlphaEdit proposal."""
    params = dict(model.named_parameters())
    missing = sorted(set(weights).difference(params))
    if missing:
        raise KeyError(f"AlphaEdit restore weights are missing from the model: {missing}")
    with torch.no_grad():
        for name, value in weights.items():
            copy_to_param(params[name], value)


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict],
    hparams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: str | None = None,
    keep_original_weight: bool = False,
    reset_cache: bool = False,
    cache_state: AlphaEditCacheState | None = None,
    defer_cache_commit: bool = False,
    return_cache_factors: bool = False,
    orig_weights_device: torch.device | str = "cpu",
    **kwargs,
) -> (
    tuple[AutoModelForCausalLM, dict[str, torch.Tensor]]
    | tuple[
        AutoModelForCausalLM,
        dict[str, torch.Tensor],
        list[torch.Tensor],
    ]
):
    """
    Apply an AlphaEdit update and optionally defer its cumulative-cache commit.

    When ``defer_cache_commit`` is true, the model receives the candidate update
    while the supplied cache state remains unchanged. Set
    ``return_cache_factors`` to receive the compact factors required by
    :func:`commit_AlphaEdit_cache_factors` after external validation. Original
    changed weights are copied to CPU by default for exact rollback without
    retaining a second copy on the GPU.
    """

    global cache_c, cache_c_key, cache_c_new

    if return_cache_factors and not defer_cache_commit:
        raise ValueError("return_cache_factors requires defer_cache_commit=True.")

    # Reset cache if requested
    if reset_cache:
        if cache_state is None:
            cache_c = None
            cache_c_new = False
            cache_c_key = None
        else:
            cache_state.reset()

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    load_AlphaEdit_projector(hparams)

    cache_matrix = _resolve_cache_matrix(model, hparams, cache_state)
    # Always defer the cache mutation until all model updates have been applied.
    # Transactional callers receive the factors and commit them after validation.
    deltas, cache_factors = execute_AlphaEdit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        cache_matrix=cache_matrix,
        defer_cache_commit=True,
    )

    prepared_updates: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for weight_name, update in deltas.items():
        weight = nethook.get_parameter(model, weight_name)
        update = upd_matrix_match_shape(update, weight.shape)
        if not torch.isfinite(update).all():
            raise ValueError(f"AlphaEdit produced a non-finite update for {weight_name}.")
        prepared_updates.append((weight_name, weight, update))

        if return_orig_weights:
            weights_copy[weight_name] = weight.detach().to(
                device=orig_weights_device,
                copy=True,
            )

    try:
        with torch.no_grad():
            for _, weight, update in prepared_updates:
                update = to_param_device(update, weight, dtype=weight.dtype)
                weight.add_(update)

        if not defer_cache_commit:
            _commit_cache_factors(cache_matrix, cache_factors)
    except Exception:
        if weights_copy:
            restore_AlphaEdit_weights(model, weights_copy)
        raise

    # print(f"New weights successfully inserted into {list(deltas.keys())}")

    if defer_cache_commit and return_cache_factors:
        return model, weights_copy, cache_factors
    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: list[dict],
    hparams,
    cache_template: str | None = None,
    cache_matrix: torch.Tensor | None = None,
    defer_cache_commit: bool = False,
) -> (
    dict[str, torch.Tensor]
    | tuple[dict[str, torch.Tensor], list[torch.Tensor]]
):
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    if cache_matrix is None:
        cache_matrix = _resolve_cache_matrix(model, hparams, None)

    deltas = {}
    cache_factors: list[torch.Tensor] = []

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
        name: weight.detach().to(device="cpu", copy=True)
        for name, weight in weights.items()
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
            cache_factors.append(layer_ks)

        if not defer_cache_commit:
            _commit_cache_factors(cache_matrix, cache_factors)
    finally:
        with torch.no_grad():
            for name, weight in weights.items():
                copy_to_param(weight, weights_copy[name])

    # print(f"Deltas successfully computed for {list(weights.keys())}")

    if defer_cache_commit:
        return deltas, cache_factors
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
) -> list[list[str]]:
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
