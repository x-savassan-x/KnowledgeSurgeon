from collections.abc import Sequence
from contextlib import contextmanager
import json
import math

import torch

from ...utils.nethook import get_hidden_state, replace_hidden_state
from .router import BgeM3Router, ExternalMemoryBank, RoutingDecision


class DecoderBlockDeltaInterceptor:
    """Adds a selected delta to the final prompt token at one block boundary."""

    def __init__(self, block: torch.nn.Module) -> None:
        self._deltas = torch.empty((0, 0), dtype=torch.float32, device="cpu")
        self._route_indices: torch.Tensor | None = None
        self._route_active: torch.Tensor | None = None
        self._direct_deltas: torch.Tensor | None = None
        self._token_indices: torch.Tensor | None = None
        self._prefill_only = True
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
        return len(self._deltas)

    @property
    def deltas(self) -> torch.Tensor:
        return self._deltas

    def clear(self) -> None:
        dimension = self._deltas.shape[1] if self._deltas.ndim == 2 else 0
        self._deltas = torch.empty(
            (0, dimension),
            dtype=self._deltas.dtype,
            device="cpu",
        )
        self.disarm()

    def append_delta(self, delta: torch.Tensor) -> int:
        values = delta.detach().to(device="cpu")
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or len(values) != 1:
            raise ValueError("append_delta expects one vector")
        if self.bank_size and values.shape[1] != self._deltas.shape[1]:
            raise ValueError(
                f"Delta dimension changed from {self._deltas.shape[1]} to {values.shape[1]}"
            )
        if not self.bank_size:
            self._deltas = values.clone()
        else:
            self._deltas = torch.cat((self._deltas, values), dim=0)
        return self.bank_size - 1

    def load_deltas(self, deltas: torch.Tensor) -> None:
        values = deltas.detach().to(device="cpu")
        if values.ndim != 2:
            raise ValueError("Delta bank must be a 2D matrix")
        self._deltas = values.clone()
        self.disarm()

    def arm_routes(
        self,
        decision: RoutingDecision,
        *,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
    ) -> None:
        self._reset_observation()
        if len(decision) == 0:
            self.disarm()
            return
        if torch.any(decision.active & (decision.top1_indices < 0)):
            raise ValueError("Active routes must have a valid delta index")
        if torch.any(decision.active & (decision.top1_indices >= self.bank_size)):
            raise IndexError("A routed delta index exceeds the interceptor bank")
        self._route_indices = decision.top1_indices.detach().to(device="cpu", dtype=torch.long)
        self._route_active = decision.active.detach().to(device="cpu", dtype=torch.bool)
        self._direct_deltas = None
        self._token_indices = self._prepare_token_indices(
            token_indices,
            len(decision),
        )
        self._prefill_only = bool(prefill_only)
        self._consumed = False

    def arm_deltas(
        self,
        deltas: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
    ) -> None:
        self._reset_observation()
        values = deltas if deltas.ndim == 2 else deltas.unsqueeze(0)
        if values.ndim != 2:
            raise ValueError("Armed deltas must be a vector or matrix")
        if active is None:
            active = torch.ones(len(values), dtype=torch.bool, device=values.device)
        if active.ndim != 1 or len(active) != len(values):
            raise ValueError("Each armed delta must have one active flag")
        self._direct_deltas = values
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
            raise ValueError("Each armed delta must have one token index")
        self._token_indices = token_indices
        self._prefill_only = bool(prefill_only)
        self._consumed = False

    def arm_memory(
        self,
        memory: ExternalMemoryBank,
        decision: RoutingDecision,
        *,
        token_indices: int | torch.Tensor = -1,
        prefill_only: bool = True,
    ) -> None:
        if len(decision) == 0:
            self.disarm()
            return
        if torch.any(decision.active & (decision.top1_indices < 0)):
            raise ValueError("Active routes must have a valid memory index")
        if torch.any(decision.active & (decision.top1_indices >= len(memory))):
            raise IndexError("A routed memory index exceeds the memory bank")
        if len(memory) == 0:
            self.disarm()
            return
        safe_indices = decision.top1_indices.clamp_min(0)
        self.arm_deltas(
            memory.deltas[safe_indices],
            active=decision.active,
            token_indices=token_indices,
            prefill_only=prefill_only,
        )

    @contextmanager
    def arm_training(
        self,
        delta: torch.Tensor,
        token_index: int | torch.Tensor,
    ):
        """Force a differentiable delta route for repeated optimization forwards."""
        self.arm_deltas(
            delta,
            token_indices=token_index,
            prefill_only=False,
        )
        try:
            yield delta
        finally:
            self.disarm()

    def disarm(self) -> None:
        self._route_indices = None
        self._route_active = None
        self._direct_deltas = None
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
        if self._direct_deltas is not None:
            deltas = self._expand_batch(self._direct_deltas, hidden.shape[0]).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            if self._route_indices is None:
                return output
            indices = self._expand_batch(self._route_indices, hidden.shape[0])
            safe_indices = indices.clamp_min(0)
            deltas = self._deltas[safe_indices].to(device=hidden.device, dtype=hidden.dtype)

        if deltas.shape[1] != hidden.shape[2]:
            raise ValueError(
                f"Delta dimension {deltas.shape[1]} does not match hidden size {hidden.shape[2]}"
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
        updated[rows, token_indices, :] += deltas * active.unsqueeze(1)
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
        router: BgeM3Router,
        interceptor: DecoderBlockDeltaInterceptor,
        memory: ExternalMemoryBank | None = None,
        diagnostics: bool = False,
    ) -> None:
        object.__setattr__(self, "base_model", model)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "router", router)
        object.__setattr__(self, "interceptor", interceptor)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "diagnostics", bool(diagnostics))
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
        memory = object.__getattribute__(self, "memory")
        diagnostics = object.__getattribute__(self, "diagnostics")
        token_indices = self._generation_token_indices(args, kwargs)
        decision = None
        if memory is None:
            decision = router.retrieve_prompts(routing_prompts)
            interceptor.arm_routes(
                decision,
                token_indices=token_indices,
                prefill_only=True,
            )
        elif len(memory) == 0:
            interceptor.disarm()
            if diagnostics:
                self._log_empty_memory(routing_prompts)
        else:
            decision = memory.route(router.encode(routing_prompts))
            interceptor.arm_memory(
                memory,
                decision,
                token_indices=token_indices,
                prefill_only=True,
            )
        try:
            output = object.__getattribute__(self, "base_model").generate(*args, **kwargs)
            if diagnostics and decision is not None:
                self._log_routing_decisions(routing_prompts, decision)
            return output
        except Exception as error:
            if diagnostics and decision is not None:
                self._log_routing_decisions(routing_prompts, decision, error=error)
            raise
        finally:
            interceptor.disarm()

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

    def _log_routing_decisions(
        self,
        prompts: list[str],
        decision: RoutingDecision,
        *,
        error: Exception | None = None,
    ) -> None:
        router = object.__getattribute__(self, "router")
        memory = object.__getattribute__(self, "memory")
        interceptor = object.__getattribute__(self, "interceptor")
        observation = interceptor.diagnostic_snapshot()
        hook_seen = int(observation["hook_calls"]) > 0

        for query_index, prompt in enumerate(prompts):
            selected_index = int(decision.top1_indices[query_index])
            active = bool(decision.active[query_index])
            score = float(decision.top1_scores[query_index])
            margin = float(decision.margins[query_index])
            below_score = score < router.score_threshold
            below_margin = margin < router.margin_threshold
            if active:
                reason = "pass"
            elif below_score and below_margin:
                reason = "score_and_margin_below_threshold"
            elif below_score:
                reason = "score_below_threshold"
            else:
                reason = "margin_below_threshold"

            selected_prompt = None
            selected_target = None
            delta_norm = None
            if memory is not None and 0 <= selected_index < len(memory):
                metadata = memory.metadata[selected_index]
                selected_prompt = getattr(metadata, "prompt", None)
                selected_target = getattr(metadata, "new_ground_truth", None)
                delta_norm = float(
                    torch.linalg.vector_norm(memory.deltas[selected_index].float())
                )

            payload = {
                "query_index": query_index,
                "query_prompt": prompt,
                "memory_size": len(memory) if memory is not None else router.bank_size,
                "selected_index": selected_index,
                "selected_prompt": selected_prompt,
                "selected_target": selected_target,
                "top1_score": self._finite_value(score),
                "score_threshold": router.score_threshold,
                "top2_score": self._finite_value(float(decision.top2_scores[query_index])),
                "margin": self._finite_value(margin),
                "margin_threshold": router.margin_threshold,
                "active": active,
                "hook_seen": hook_seen,
                "intercepted": active and hook_seen,
                "delta_norm": delta_norm,
                "reason": reason,
            }
            if error is not None:
                payload["generation_error"] = f"{type(error).__name__}: {error}"
            self._emit_diagnostic(payload)

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
