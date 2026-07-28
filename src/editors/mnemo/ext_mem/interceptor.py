from collections.abc import Sequence
from contextlib import contextmanager
import json
import math
import operator

import torch

from ....utils.nethook import get_hidden_state, replace_hidden_state
from .router import Router, RoutingDecision
from .storage import MnemoStorage, PatchStore


class Interceptor:
    """Applies a routed target state at one decoder-block boundary."""

    def __init__(self, block: torch.nn.Module) -> None:
        self._values = torch.empty((0, 0), dtype=torch.float32, device="cpu")
        self._route_indices: torch.Tensor | None = None
        self._route_active: torch.Tensor | None = None
        self._direct_values: torch.Tensor | None = None
        self._token_indices: torch.Tensor | None = None
        self._prefill_only = True
        self._intervention_mode = "replace"
        self._strength = 1.0
        self._consumed = False
        self._armed_hook_calls = 0
        self._applied_rows = torch.empty(0, dtype=torch.bool)
        self._applied_token_indices = torch.empty(0, dtype=torch.long)
        try:
            self._hook_handle = block.register_forward_hook(self._hook, with_kwargs=True)
        except TypeError:
            self._hook_handle = block.register_forward_hook(self._legacy_hook)

    @property
    def bank_size(self) -> int:
        return len(self._values)

    @property
    def values(self) -> torch.Tensor:
        return self._values

    def clear(self) -> None:
        dimension = self._values.shape[1] if self._values.ndim == 2 else 0
        self._values = torch.empty(
            (0, dimension),
            dtype=self._values.dtype,
            device="cpu",
        )
        self.disarm()

    def append_value(self, value: torch.Tensor) -> int:
        values = value.detach().to(device="cpu")
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or len(values) != 1:
            raise ValueError("append_value expects one vector")
        if self.bank_size and values.shape[1] != self._values.shape[1]:
            raise ValueError(
                f"Value dimension changed from {self._values.shape[1]} to {values.shape[1]}"
            )
        if not self.bank_size:
            self._values = values.clone()
        else:
            self._values = torch.cat((self._values, values), dim=0)
        return self.bank_size - 1

    def load_values(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu")
        if values.ndim != 2:
            raise ValueError("Value bank must be a 2D matrix")
        self._values = values.clone()
        self.disarm()

    def arm_routes(
        self,
        decision: RoutingDecision,
        *,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
        strength: float = 1.0,
    ) -> None:
        self._reset_observation()
        if len(decision) == 0:
            self.disarm()
            return
        if torch.any(decision.active & (decision.top1_indices < 0)):
            raise ValueError("Active routes must have a valid value index")
        if torch.any(decision.active & (decision.top1_indices >= self.bank_size)):
            raise IndexError("A routed value index exceeds the interceptor bank")
        self._route_indices = decision.top1_indices.detach().to(device="cpu", dtype=torch.long)
        self._route_active = decision.active.detach().to(device="cpu", dtype=torch.bool)
        self._direct_values = None
        self._token_indices = self._prepare_token_indices(
            token_indices,
            len(decision),
        )
        self._prefill_only = bool(prefill_only)
        self._intervention_mode = "replace"
        if not 0.0 < strength <= 1.0:
            raise ValueError("Replacement strength must be in (0, 1].")
        self._strength = float(strength)
        self._consumed = False

    def arm_values(
        self,
        values: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
        strength: float = 1.0,
    ) -> None:
        self._reset_observation()
        values = values if values.ndim == 2 else values.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("Armed values must be a vector or matrix")
        if not 0.0 < strength <= 1.0:
            raise ValueError("Replacement strength must be in (0, 1].")
        if active is None:
            active = torch.ones(len(values), dtype=torch.bool, device=values.device)
        if active.ndim != 1 or len(active) != len(values):
            raise ValueError("Each armed value must have one active flag")
        self._direct_values = values
        self._route_indices = None
        self._route_active = active
        if isinstance(token_indices, int):
            token_indices = torch.full(
                (len(values),),
                token_indices,
                dtype=torch.long,
                device=values.device,
            )
        if token_indices.ndim != 1 or len(token_indices) != len(values):
            raise ValueError("Each armed value must have one token index")
        self._token_indices = token_indices
        self._prefill_only = bool(prefill_only)
        self._intervention_mode = "replace"
        self._strength = float(strength)
        self._consumed = False

    def arm_patches(
        self,
        patches: PatchStore,
        decision: RoutingDecision,
        *,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
        strength: float = 1.0,
    ) -> None:
        if len(decision) == 0:
            self.disarm()
            return
        if torch.any(decision.active & (decision.top1_indices < 0)):
            raise ValueError("Active routes must have a valid patch index")
        if torch.any(decision.active & (decision.top1_indices >= len(patches))):
            raise IndexError("A routed patch index exceeds the patch store")
        if len(patches) == 0:
            self.disarm()
            return
        safe_indices = decision.top1_indices.clamp_min(0)
        self.arm_values(
            patches.values[safe_indices],
            active=decision.active,
            token_indices=token_indices,
            prefill_only=prefill_only,
            strength=strength,
        )

    @contextmanager
    def arm_training_delta(
        self,
        delta: torch.Tensor,
        token_index: int | torch.Tensor,
    ):
        """Force a differentiable delta route for repeated optimization forwards."""
        self._reset_observation()
        values = delta if delta.ndim == 2 else delta.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("Training delta must be a vector or matrix")
        self._direct_values = values
        self._route_indices = None
        self._route_active = torch.ones(
            len(values),
            dtype=torch.bool,
            device=values.device,
        )
        self._token_indices = self._prepare_token_indices(token_index, len(values))
        self._prefill_only = False
        self._intervention_mode = "add"
        self._strength = 1.0
        self._consumed = False
        try:
            yield delta
        finally:
            self.disarm()

    def disarm(self) -> None:
        self._route_indices = None
        self._route_active = None
        self._direct_values = None
        self._token_indices = None
        self._consumed = False

    def close(self) -> None:
        self.disarm()
        self._hook_handle.remove()

    def diagnostic_snapshot(self) -> dict[str, object]:
        return {
            "hook_calls": self._armed_hook_calls,
            "applied_rows": self._applied_rows.tolist(),
            "token_indices": self._applied_token_indices.tolist(),
        }

    def _reset_observation(self) -> None:
        self._armed_hook_calls = 0
        self._applied_rows = torch.empty(0, dtype=torch.bool)
        self._applied_token_indices = torch.empty(0, dtype=torch.long)

    def _legacy_hook(self, module, inputs, output):
        return self._hook(module, inputs, {}, output)

    def _hook(self, _module, _inputs, _kwargs, output):
        if self._route_active is None:
            return output
        self._armed_hook_calls += 1
        if self._prefill_only and self._consumed:
            return output

        hidden = get_hidden_state(output)
        if hidden.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, hidden], got {tuple(hidden.shape)}")

        active = self._expand_batch(self._route_active, hidden.shape[0]).to(
            device=hidden.device,
            dtype=torch.bool,
        )
        if self._direct_values is not None:
            values = self._expand_batch(self._direct_values, hidden.shape[0]).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            if self._route_indices is None:
                return output
            indices = self._expand_batch(self._route_indices, hidden.shape[0])
            safe_indices = indices.clamp_min(0)
            values = self._values[safe_indices].to(device=hidden.device, dtype=hidden.dtype)

        if values.shape[1] != hidden.shape[2]:
            raise ValueError(
                f"Value dimension {values.shape[1]} does not match hidden size {hidden.shape[2]}"
            )

        self._consumed = True
        if not torch.any(active):
            self._applied_rows = active.detach().to(device="cpu")
            return output

        if self._token_indices is None:
            token_indices = torch.full(
                (hidden.shape[0],),
                -1,
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            token_indices = self._expand_batch(self._token_indices, hidden.shape[0]).to(
                device=hidden.device,
                dtype=torch.long,
            )
        token_indices = torch.where(
            token_indices < 0,
            token_indices + hidden.shape[1],
            token_indices,
        )
        if torch.any((token_indices < 0) | (token_indices >= hidden.shape[1])):
            raise IndexError("An interceptor token index is outside the sequence")
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        self._applied_rows = active.detach().to(device="cpu")
        self._applied_token_indices = token_indices.detach().to(device="cpu")
        updated = hidden.clone()
        current = hidden[rows, token_indices, :]
        if self._intervention_mode == "add":
            intervened = current + values
        else:
            intervened = current + self._strength * (values - current)
        updated[rows, token_indices, :] = torch.where(
            active.unsqueeze(1),
            intervened,
            current,
        )
        return replace_hidden_state(output, updated)

    @staticmethod
    def _expand_batch(values: torch.Tensor, target_size: int) -> torch.Tensor:
        source_size = len(values)
        if source_size == target_size:
            return values
        if source_size == 0 or target_size % source_size:
            raise ValueError(
                f"Cannot expand routing batch of {source_size} to model batch of {target_size}"
            )
        return values.repeat_interleave(target_size // source_size, dim=0)

    @staticmethod
    def _prepare_token_indices(
        token_indices: int | torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if isinstance(token_indices, int):
            return torch.full((batch_size,), token_indices, dtype=torch.long)
        if token_indices.ndim != 1 or len(token_indices) != batch_size:
            raise ValueError("Each route must have one token index")
        return token_indices.detach().to(device="cpu", dtype=torch.long)


class PrefillRoutingModel:
    """Proxy that routes raw prompts once before delegated autoregressive generation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        router: Router,
        interceptor: Interceptor,
        storage: MnemoStorage | None = None,
        router_diagnostics: bool = False,
        interceptor_diagnostics: bool = False,
        intervention_strength: float = 1.0,
    ) -> None:
        object.__setattr__(self, "base_model", model)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "router", router)
        object.__setattr__(self, "interceptor", interceptor)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "router_diagnostics", bool(router_diagnostics))
        object.__setattr__(
            self,
            "interceptor_diagnostics",
            bool(interceptor_diagnostics),
        )
        object.__setattr__(
            self,
            "intervention_strength",
            float(intervention_strength),
        )
        object.__setattr__(self, "_forced_patch_indices", None)
        object.__setattr__(self, "_forced_patch_strength", None)
        object.__setattr__(self, "accepts_routing_prompts", True)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "base_model"), name)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "base_model")(*args, **kwargs)

    def generate(self, *args, **kwargs):
        routing_prompts = kwargs.pop("routing_prompts", None)
        if routing_prompts is None:
            routing_prompts = self._decode_generation_prompts(args, kwargs)
        else:
            routing_prompts = list(routing_prompts)

        router = object.__getattribute__(self, "router")
        interceptor = object.__getattribute__(self, "interceptor")
        storage = object.__getattribute__(self, "storage")
        router_diagnostics = object.__getattribute__(self, "router_diagnostics")
        interceptor_diagnostics = object.__getattribute__(
            self, "interceptor_diagnostics"
        )
        token_indices = self._generation_token_indices(args, kwargs)
        decision = None
        forced_patch_indices = object.__getattribute__(
            self, "_forced_patch_indices"
        )
        forced_patch_strength = object.__getattribute__(
            self, "_forced_patch_strength"
        )
        if forced_patch_indices is not None:
            if storage is None or len(storage) == 0:
                raise RuntimeError("Cannot force a patch with an empty memory store")
            decision = self._forced_decision(
                forced_patch_indices,
                len(routing_prompts),
                len(storage),
            )
            interceptor.arm_patches(
                storage.patches,
                decision,
                token_indices=token_indices,
                prefill_only=True,
                strength=(
                    object.__getattribute__(self, "intervention_strength")
                    if forced_patch_strength is None
                    else forced_patch_strength
                ),
            )
        elif storage is None:
            decision = router.retrieve_prompts(routing_prompts)
            interceptor.arm_routes(
                decision,
                token_indices=token_indices,
                prefill_only=True,
                strength=object.__getattribute__(self, "intervention_strength"),
            )
        elif len(storage) == 0:
            interceptor.disarm()
            if router_diagnostics:
                self._log_empty_memory(routing_prompts)
        else:
            decision = storage.route(router.encode(routing_prompts))
            interceptor.arm_patches(
                storage.patches,
                decision,
                token_indices=token_indices,
                prefill_only=True,
                strength=object.__getattribute__(self, "intervention_strength"),
            )
        try:
            output = object.__getattribute__(self, "base_model").generate(*args, **kwargs)
            if (router_diagnostics or interceptor_diagnostics) and decision is not None:
                self._log_generation_diagnostics(
                    routing_prompts,
                    decision,
                    emit_router=router_diagnostics,
                    emit_interceptor=interceptor_diagnostics,
                    forced=forced_patch_indices is not None,
                    applied_strength=(
                        forced_patch_strength
                        if forced_patch_indices is not None
                        and forced_patch_strength is not None
                        else object.__getattribute__(self, "intervention_strength")
                    ),
                )
            return output
        except Exception as error:
            if (router_diagnostics or interceptor_diagnostics) and decision is not None:
                self._log_generation_diagnostics(
                    routing_prompts,
                    decision,
                    emit_router=router_diagnostics,
                    emit_interceptor=interceptor_diagnostics,
                    forced=forced_patch_indices is not None,
                    applied_strength=(
                        forced_patch_strength
                        if forced_patch_indices is not None
                        and forced_patch_strength is not None
                        else object.__getattribute__(self, "intervention_strength")
                    ),
                    error=error,
                )
            raise
        finally:
            interceptor.disarm()

    @contextmanager
    def force_patch(
        self,
        patch_indices: int | Sequence[int],
        *,
        strength: float | None = None,
    ):
        """Temporarily bypass routing and apply selected stored patch rows."""
        if object.__getattribute__(self, "_forced_patch_indices") is not None:
            raise RuntimeError("Nested forced-patch contexts are not supported")
        normalized = self._normalize_forced_indices(patch_indices)
        if strength is not None and not 0.0 < float(strength) <= 1.0:
            raise ValueError("Forced replacement strength must be in (0, 1].")
        object.__setattr__(self, "_forced_patch_indices", normalized)
        object.__setattr__(
            self,
            "_forced_patch_strength",
            None if strength is None else float(strength),
        )
        try:
            yield
        finally:
            object.__setattr__(self, "_forced_patch_indices", None)
            object.__setattr__(self, "_forced_patch_strength", None)

    @staticmethod
    def _normalize_forced_indices(
        patch_indices: int | Sequence[int],
    ) -> int | tuple[int, ...]:
        if isinstance(patch_indices, bool):
            raise TypeError("Forced patch index must be an integer")
        if isinstance(patch_indices, int):
            return patch_indices
        normalized = []
        for index in patch_indices:
            if isinstance(index, bool):
                raise TypeError("Forced patch indices must be integers")
            try:
                normalized.append(operator.index(index))
            except TypeError as error:
                raise TypeError("Forced patch indices must be integers") from error
        indices = tuple(normalized)
        if not indices:
            raise ValueError("At least one forced patch index is required")
        return indices

    @staticmethod
    def _forced_decision(
        patch_indices: int | tuple[int, ...],
        prompt_count: int,
        bank_size: int,
    ) -> RoutingDecision:
        if isinstance(patch_indices, int):
            indices = torch.full(
                (prompt_count,),
                patch_indices,
                dtype=torch.long,
            )
        else:
            if len(patch_indices) != prompt_count:
                raise ValueError(
                    "A forced patch sequence must align with the prompt batch"
                )
            indices = torch.tensor(patch_indices, dtype=torch.long)
        if torch.any((indices < 0) | (indices >= bank_size)):
            raise IndexError("A forced patch index exceeds the memory store")
        return RoutingDecision(
            top1_indices=indices,
            top1_scores=torch.full((prompt_count,), torch.inf),
            top2_scores=torch.full((prompt_count,), -torch.inf),
            margins=torch.full((prompt_count,), torch.inf),
            active=torch.ones(prompt_count, dtype=torch.bool),
        )

    def _log_empty_memory(self, prompts: list[str]) -> None:
        for query_index, prompt in enumerate(prompts):
            self._emit_diagnostic({
                "query_index": query_index,
                "query_prompt": prompt,
                "memory_size": 0,
                "active": False,
                "intercepted": False,
                "reason": "memory_empty",
            })

    def _log_generation_diagnostics(
        self,
        prompts: list[str],
        decision: RoutingDecision,
        *,
        emit_router: bool,
        emit_interceptor: bool,
        forced: bool = False,
        applied_strength: float | None = None,
        error: Exception | None = None,
    ) -> None:
        router = object.__getattribute__(self, "router")
        storage = object.__getattribute__(self, "storage")
        interceptor = object.__getattribute__(self, "interceptor")
        observation = interceptor.diagnostic_snapshot()
        hook_seen = int(observation["hook_calls"]) > 0

        if emit_router:
            for query_index, prompt in enumerate(prompts):
                selected_index = int(decision.top1_indices[query_index])
                active = bool(decision.active[query_index])
                score = float(decision.top1_scores[query_index])
                margin = float(decision.margins[query_index])
                below_score = score < router.score_threshold
                below_margin = margin < router.margin_threshold
                if forced:
                    reason = "forced_patch"
                elif active:
                    reason = "pass"
                elif below_score and below_margin:
                    reason = "score_and_margin_below_threshold"
                elif below_score:
                    reason = "score_below_threshold"
                else:
                    reason = "margin_below_threshold"

                selected_prompt = None
                selected_subject = None
                selected_answer = None
                value_norm = None
                if storage is not None and 0 <= selected_index < len(storage):
                    selected_subject = storage.edits.subject_at(selected_index)
                    selected_prompt = storage.edits.prompt_at(selected_index)
                    selected_answer = storage.edits.answer_at(selected_index)
                    value_norm = float(
                        torch.linalg.vector_norm(storage.values[selected_index].float())
                    )

                payload = {
                    "query_index": query_index,
                    "query_prompt": prompt,
                    "memory_size": len(storage) if storage is not None else router.bank_size,
                    "selected_index": selected_index,
                    "selected_subject": selected_subject,
                    "selected_prompt": selected_prompt,
                    "selected_answer": selected_answer,
                    "top1_score": self._finite_value(score),
                    "score_threshold": router.score_threshold,
                    "top2_score": self._finite_value(
                        float(decision.top2_scores[query_index])
                    ),
                    "margin": self._finite_value(margin),
                    "margin_threshold": router.margin_threshold,
                    "active": active,
                    "hook_seen": hook_seen,
                    "intercepted": active and hook_seen,
                    "value_norm": value_norm,
                    "intervention_strength": (
                        object.__getattribute__(self, "intervention_strength")
                        if applied_strength is None
                        else applied_strength
                    ),
                    "reason": reason,
                    "forced": forced,
                }
                if error is not None:
                    payload["generation_error"] = f"{type(error).__name__}: {error}"
                self._emit_diagnostic(payload)

        if emit_interceptor:
            self._log_interceptor_diagnostics(observation)

    def _log_interceptor_diagnostics(
        self,
        observation: dict[str, object] | None = None,
    ) -> None:
        if observation is None:
            observation = object.__getattribute__(
                self, "interceptor"
            ).diagnostic_snapshot()
        print(
            "[mnemo-hook] "
            + json.dumps(observation, ensure_ascii=True, sort_keys=True),
            flush=True,
        )

    @staticmethod
    def _finite_value(value: float) -> float | str:
        if math.isfinite(value):
            return value
        return "inf" if value > 0 else "-inf"

    @staticmethod
    def _emit_diagnostic(payload: dict[str, object]) -> None:
        print(
            "[mnemo-route] "
            + json.dumps(payload, ensure_ascii=True, sort_keys=True),
            flush=True,
        )

    def _decode_generation_prompts(self, args: Sequence, kwargs: dict) -> list[str]:
        input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Generation routing requires a 2D input_ids tensor")

        attention_mask = kwargs.get("attention_mask")
        rows = []
        for index, row in enumerate(input_ids):
            if attention_mask is not None:
                row = row[attention_mask[index].to(dtype=torch.bool, device=row.device)]
            rows.append(row.detach().cpu())
        return object.__getattribute__(self, "tokenizer").batch_decode(
            rows,
            skip_special_tokens=True,
        )

    @staticmethod
    def _generation_token_indices(args: Sequence, kwargs: dict) -> torch.Tensor:
        input_ids = kwargs.get("input_ids", kwargs.get("inputs"))
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Generation routing requires a 2D input_ids tensor")
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            return torch.full((len(input_ids),), input_ids.shape[1] - 1, dtype=torch.long)
        indices = []
        for mask in attention_mask:
            active = torch.nonzero(mask, as_tuple=False)
            if not len(active):
                raise ValueError("Generation prompt attention masks must not be empty")
            indices.append(int(active[-1, 0]))
        return torch.tensor(indices, dtype=torch.long)
