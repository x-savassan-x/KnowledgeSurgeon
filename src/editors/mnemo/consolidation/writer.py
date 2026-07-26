from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .alpha_edit import (
    AlphaEditCacheState,
    _resolve_cache_matrix,
    commit_AlphaEdit_cache_factors,
    execute_AlphaEdit,
    load_AlphaEdit_projector,
    release_AlphaEdit_projector,
    upd_matrix_match_shape,
)
from ....utils import nethook
from ....utils.devices import copy_to_param
from ...editor_utils import EditRequest


_UPDATE_CHUNK_ELEMENTS = 1_000_000


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), _UPDATE_CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[start:start + _UPDATE_CHUNK_ELEMENTS]).all()):
            return False
    return True


def _tensor_statistics(tensor: torch.Tensor) -> tuple[bool, float, float]:
    flat = tensor.detach().reshape(-1)
    norm_sq = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, flat.numel(), _UPDATE_CHUNK_ELEMENTS):
        chunk = flat[start:start + _UPDATE_CHUNK_ELEMENTS].float()
        finite = finite and bool(torch.isfinite(chunk).all())
        norm_sq += float(torch.sum(chunk * chunk).cpu())
        if chunk.numel():
            max_abs = max(max_abs, float(chunk.abs().max().cpu()))
        del chunk
    return finite, norm_sq ** 0.5, max_abs


def _backend_request(request: EditRequest) -> dict[str, str]:
    return {
        "prompt": request.prompt,
        "subject": request.subject,
        "target_new": request.answer,
    }


@dataclass
class ConsolidationProposal:
    """A deferred projected update that can be committed or rolled back exactly."""

    model: AutoModelForCausalLM
    cache_state: AlphaEditCacheState
    deltas: dict[str, torch.Tensor]
    cache_factors: list[torch.Tensor]
    _snapshots: dict[str, torch.Tensor] = field(default_factory=dict, init=False)
    _applied: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def applied(self) -> bool:
        return self._applied

    @property
    def snapshots(self) -> dict[str, torch.Tensor]:
        if not self._applied:
            raise RuntimeError("AlphaEdit proposal snapshots exist only while applied.")
        return self._snapshots

    def statistics(self) -> dict[str, dict[str, float | bool]]:
        params = dict(self.model.named_parameters())
        stats: dict[str, dict[str, float | bool]] = {}
        for name, delta in self.deltas.items():
            parameter_finite, parameter_norm, _ = _tensor_statistics(params[name])
            update_finite, update_norm, max_abs_update = _tensor_statistics(delta)
            stats[name] = {
                "finite": parameter_finite and update_finite,
                "parameter_norm": parameter_norm,
                "update_norm": update_norm,
                "relative_update_norm": update_norm / max(parameter_norm, 1e-12),
                "max_abs_update": max_abs_update,
            }
        return stats

    def apply(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot apply a closed AlphaEdit proposal.")
        if self._applied:
            return

        params = dict(self.model.named_parameters())
        prepared_updates: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, delta in self.deltas.items():
            if name not in params:
                raise KeyError(f"AlphaEdit proposal parameter is missing: {name}")
            parameter = params[name]
            update = upd_matrix_match_shape(delta, parameter.shape)
            if not _tensor_is_finite(parameter) or not _tensor_is_finite(update):
                raise FloatingPointError(
                    f"AlphaEdit proposal contains non-finite values for {name}."
                )
            prepared_updates.append((name, parameter, update))

        self._snapshots = {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter, _ in prepared_updates
        }
        try:
            with torch.no_grad():
                for _, parameter, update in prepared_updates:
                    rows_per_chunk = max(
                        1,
                        _UPDATE_CHUNK_ELEMENTS // max(1, parameter.shape[-1]),
                    )
                    for row_start in range(0, parameter.shape[0], rows_per_chunk):
                        row_end = min(parameter.shape[0], row_start + rows_per_chunk)
                        update_chunk = update[row_start:row_end].to(
                            device=parameter.device,
                            dtype=torch.float32,
                        )
                        updated = parameter[row_start:row_end].float() + update_chunk
                        copy_to_param(parameter[row_start:row_end], updated)
                        del update_chunk, updated
            self._applied = True
        except BaseException:
            self._restore_snapshots(params)
            self._snapshots.clear()
            raise

    def commit(self) -> None:
        if self._closed:
            raise RuntimeError("AlphaEdit proposal is already closed.")
        if not self._applied:
            raise RuntimeError("Apply the AlphaEdit proposal before committing it.")
        try:
            commit_AlphaEdit_cache_factors(self.cache_state, self.cache_factors)
        except BaseException:
            self.rollback()
            raise
        self._snapshots.clear()
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        if self._applied:
            self._restore_snapshots(dict(self.model.named_parameters()))
        self._snapshots.clear()
        self._applied = False
        self._closed = True

    def _restore_snapshots(self, params: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, snapshot in self._snapshots.items():
                copy_to_param(params[name], snapshot)


class ConsolidationWriter:
    """Owns Mnemo's private consolidation cache and reset state."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.cache_state = AlphaEditCacheState()
        self.weight_names = [
            f"{config.rewrite_module_tmp.format(layer)}.weight"
            for layer in config.layers
        ]
        params = dict(model.named_parameters())
        self._origin_weights = {
            name: params[name].detach().to(device="cpu", copy=True)
            for name in self.weight_names
        }

    @property
    def origin_weights(self) -> dict[str, torch.Tensor]:
        return self._origin_weights

    def propose(self, requests: list[EditRequest]) -> ConsolidationProposal:
        if not requests:
            raise ValueError("Cannot consolidate an empty request batch.")
        load_AlphaEdit_projector(self.config)
        cache_matrix = _resolve_cache_matrix(
            self.model,
            self.config,
            self.cache_state,
        )
        deltas, cache_factors = execute_AlphaEdit(
            self.model,
            self.tokenizer,
            [_backend_request(request) for request in requests],
            self.config,
            cache_matrix=cache_matrix,
            defer_cache_commit=True,
        )
        return ConsolidationProposal(
            model=self.model,
            cache_state=self.cache_state,
            deltas=deltas,
            cache_factors=cache_factors,
        )

    def release_consolidation_state(self) -> None:
        self.cache_state.reset()
        release_AlphaEdit_projector(self.config.P_loc)

    def reset(self) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._origin_weights.items():
                copy_to_param(params[name], weight)
        self.cache_state.reset()
