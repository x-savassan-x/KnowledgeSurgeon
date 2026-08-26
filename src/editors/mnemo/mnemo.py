import gc
import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
from math import isfinite
from pathlib import Path

import torch

from ...config import MnemoConfig
from ...utils import nethook
from ...utils.devices import get_model_device
from ..editor import Editor
from ..editor_utils import EditRequest
from .consolidation import (
    ConsolidationProposal,
    ConsolidationValidator,
    ConsolidationWriter,
)
from .ext_mem.interceptor import Interceptor, PrefillRoutingModel
from .ext_mem.router import build_router
from .ext_mem.storage import MnemoStorage
from .ext_mem.value import InterventionTrainer
from .safety import (
    ConsolidationSafetyResult,
    NumericalGuardResult,
    consolidation_safety_summary,
)


class Mnemo(Editor):
    """Mnemo editor supporting hybrid and external_only operating modes."""

    config: MnemoConfig

    def __init__(self, config: MnemoConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self._base_model = self.model
        self.device = get_model_device(self._base_model, fallback=config.device)
        self.config.device = self.device
        self.config.consolidation_config.device = self.device
        self.config.consolidation_config.model_key = config.model_key
        if not config.external_only:
            projector_path = Path(config.consolidation_config.P_loc)
            if not projector_path.exists():
                raise FileNotFoundError(
                    f"Mnemo requires the AlphaEdit projector: {projector_path}. "
                    "Download it with notebook/assets.ipynb before editing."
                )
        self._original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        self._original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self._base_model.named_parameters()
        }
        self._freeze_base_model()

        router_config = config.router_config
        router_device = router_config.device
        if router_device == "auto":
            router_device = (
                f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
            )
        routing_profile = router_config.profile_for(0)
        score_threshold = routing_profile.score_threshold
        margin_threshold = routing_profile.margin_threshold
        self.router = build_router(
            backend=router_config.backend,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            model_name=router_config.model_name,
            device=router_device,
            use_fp16=router_config.fp16,
            max_length=router_config.max_length,
            batch_size=router_config.batch_size,
        )
        if router_config.eager_load:
            self.router.load()
        self.storage = MnemoStorage(
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            compact_edits=config.external_only,
        )
        self.writer = (
            None
            if config.external_only
            else ConsolidationWriter(
                self._base_model,
                self.tokenizer,
                config.consolidation_config,
            )
        )

        self._interceptor: Interceptor | None = None
        self._value_trainer: InterventionTrainer | None = None
        self._validator = self._new_validator() if not config.external_only else None
        self._consolidation_count = 0
        self._consolidated_edit_count = 0
        self._external_only = config.external_only
        self._routing_profile_max_size = routing_profile.max_bank_size
        self._closed = True
        self.last_delta_loss: float | None = None
        self.last_delta_norm: float | None = None
        self.last_delta_saturated: bool | None = None
        self.last_consolidation_report: dict | None = None
        self.consolidation_history: list[dict] = []
        self._pending_log_events: list[dict[str, object]] = []
        self._install_runtime()
        if self.config.safety_config.enabled and not self._external_only:
            # Preserve the pretrained reference before deferred consolidations begin.
            self._require_validator().ensure_reference()

    def edit(self, request: EditRequest) -> None:
        self._install_runtime()
        key = self.router.encode_keys([request.prompt])
        value, delta_norm, loss = self._require_value_trainer().train(request)
        memory_index = self.storage.append(
            key,
            value,
            request=request,
        )
        self._refresh_routing_thresholds()
        self.last_delta_loss = loss
        self.last_delta_norm = delta_norm
        max_norm = self.config.delta_config.max_norm
        self.last_delta_saturated = self.last_delta_norm >= max_norm * (1.0 - 1.0e-4)
        if self.config.diagnostics_config.delta_enabled:
            payload = {
                "event": "edit",
                "memory_index": memory_index,
                "prompt": request.prompt,
                "answer": request.answer,
                "final_loss": self.last_delta_loss,
                "delta_norm": self.last_delta_norm,
                "max_norm": max_norm,
                "max_norm_reached": self.last_delta_saturated,
            }
            self._pending_log_events.append(payload)
            print("[mnemo-edit] " + json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)

        if (
            not self._external_only
            and self.buffer_size >= self.config.consolidation_cycle
        ):
            self.consolidate()

    def reset(self) -> None:
        if self.writer is not None:
            self.writer.reset()
        self.storage.reset(compact_edits=self.config.external_only)
        self.router.clear()
        if self._validator is not None:
            self._validator.reset()
        self._consolidation_count = 0
        self._consolidated_edit_count = 0
        self._external_only = self.config.external_only
        self._refresh_routing_thresholds(force=True)
        self.last_delta_loss = None
        self.last_delta_norm = None
        self.last_delta_saturated = None
        self.last_consolidation_report = None
        self.consolidation_history.clear()
        self._pending_log_events.clear()
        if self._interceptor is not None:
            self._interceptor.close()
            self._interceptor = None
        self._value_trainer = None
        self.model = self._base_model
        self._closed = True
        self.tokenizer.padding_side = self._original_padding_side
        for name, parameter in self._base_model.named_parameters():
            parameter.requires_grad_(self._original_requires_grad[name])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def buffer_size(self) -> int:
        return 0 if self._external_only else len(self.storage.edits)

    @property
    def external_only(self) -> bool:
        return self._external_only

    def drain_log_events(self) -> list[dict[str, object]]:
        events = self._pending_log_events
        self._pending_log_events = []
        if self._interceptor is not None:
            events.extend(self._interceptor.drain_diagnostic_events())
        return events

    @property
    def external_memory_size(self) -> int:
        return len(self.storage)

    @contextmanager
    def force_patch(
        self,
        patch_indices: int | Sequence[int],
        *,
        strength: float | None = None,
    ):
        """Temporarily bypass Mnemo routing for diagnostic intervention tests."""
        self._install_runtime()
        if not isinstance(self.model, PrefillRoutingModel):
            raise RuntimeError("Mnemo runtime does not expose forced patch routing")
        with self.model.force_patch(patch_indices, strength=strength):
            yield

    def consolidate(self) -> bool:
        if self._external_only or not len(self.storage):
            return False

        requests = self.storage.edits.records()
        self._require_interceptor().disarm()
        writer = self._require_writer()
        try:
            proposal = writer.propose(requests)
        except Exception as error:
            self._enter_external_only("proposal_error", error=error)
            return False

        safety_config = self.config.safety_config
        numerical = self._proposal_numerical_guard(proposal)
        if not numerical.safe:
            proposal.rollback()
            self._enter_external_only(
                "numerical",
                metrics={
                    "safe": numerical.safe,
                    "warning": numerical.warning,
                    "maximum_relative_update_norm": (
                        numerical.maximum_relative_update_norm
                        if isfinite(numerical.maximum_relative_update_norm)
                        else None
                    ),
                    "warning_limit": safety_config.relative_update_norm_warning,
                    "hard_limit": safety_config.relative_update_norm_hard_limit,
                    "nonfinite_tensors": list(numerical.nonfinite_tensors),
                },
            )
            return False

        if not safety_config.enabled:
            safe = self._commit_without_full_safety(
                proposal,
                requests,
                safety_status="disabled",
            )
        elif self._consolidated_edit_count < safety_config.start_after_edits:
            try:
                self._require_validator().ensure_reference()
            except Exception as error:
                proposal.rollback()
                self._enter_external_only("validation_error", error=error)
                return False
            safe = self._commit_without_full_safety(
                proposal,
                requests,
                safety_status="deferred",
            )
        else:
            safe = self._validate_and_finish(proposal, requests, numerical)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return safe

    def _install_runtime(self) -> None:
        if not self._closed:
            return
        if self.writer is None and not self.config.external_only:
            projector_path = Path(self.config.consolidation_config.P_loc)
            if not projector_path.exists():
                raise FileNotFoundError(
                    f"Mnemo requires the AlphaEdit projector: {projector_path}. "
                    "Download it with notebook/assets.ipynb before editing."
                )
            self.writer = ConsolidationWriter(
                self._base_model,
                self.tokenizer,
                self.config.consolidation_config,
            )
        if self._validator is None and not self._external_only:
            self._validator = self._new_validator()
        self._freeze_base_model()
        self.tokenizer.padding_side = "right"
        block_name = self.config.intervention_module_tmp.format(
            self.config.intervention_block
        )
        block = nethook.get_module(self._base_model, block_name)
        self._interceptor = Interceptor(block)
        self._value_trainer = InterventionTrainer(
            self._base_model,
            self.tokenizer,
            self._interceptor,
            self.config.delta_config,
            module_name=block_name,
            device=self.device,
        )
        self.model = PrefillRoutingModel(
            self._base_model,
            self.tokenizer,
            self.router,
            self._interceptor,
            storage=self.storage,
            router_diagnostics=self.config.diagnostics_config.router_enabled,
            interceptor_diagnostics=self.config.diagnostics_config.interceptor_enabled,
            intervention_strength=self.config.intervention_strength,
        )
        self._closed = False

    def _freeze_base_model(self) -> None:
        for parameter in self._base_model.parameters():
            parameter.requires_grad_(False)

    def _require_interceptor(self) -> Interceptor:
        if self._interceptor is None:
            raise RuntimeError("Mnemo interceptor is not installed.")
        return self._interceptor

    def set_intervention_strength(self, strength: float) -> None:
        """Update the generation proxy's intervention strength at runtime."""
        strength = float(strength)
        if not 0.0 < strength <= 1.0:
            raise ValueError("Mnemo intervention_strength must be in (0, 1].")
        runtime = self.model
        if not isinstance(runtime, PrefillRoutingModel):
            raise RuntimeError("Mnemo generation runtime is not installed.")
        runtime.intervention_strength = strength

    @property
    def intervention_strength(self) -> float:
        runtime = self.model
        if not isinstance(runtime, PrefillRoutingModel):
            return float(self.config.intervention_strength)
        return float(runtime.intervention_strength)

    def _require_writer(self) -> ConsolidationWriter:
        if self.writer is None:
            raise RuntimeError(
                "Mnemo has no consolidation writer in external_only mode."
            )
        return self.writer

    def _require_value_trainer(self) -> InterventionTrainer:
        if self._value_trainer is None:
            raise RuntimeError("Mnemo intervention trainer is not installed.")
        return self._value_trainer

    def _require_validator(self) -> ConsolidationValidator:
        if self._validator is None:
            raise RuntimeError(
                "Mnemo has no consolidation validator in external_only mode."
            )
        return self._validator

    def _new_validator(self) -> ConsolidationValidator:
        return ConsolidationValidator(
            self._base_model,
            self.tokenizer,
            self.config.safety_config,
            self.config.delta_config,
            device=self.device,
            lm_head_module=self.config.consolidation_config.lm_head_module,
        )

    def _refresh_routing_thresholds(self, *, force: bool = False) -> None:
        profile = self.config.router_config.profile_for(self.external_memory_size)
        if not force and profile.max_bank_size == self._routing_profile_max_size:
            return
        self.router.set_thresholds(
            profile.score_threshold,
            profile.margin_threshold,
        )
        self.storage.set_thresholds(
            profile.score_threshold,
            profile.margin_threshold,
        )
        self._routing_profile_max_size = profile.max_bank_size

    def _proposal_numerical_guard(
        self,
        proposal: ConsolidationProposal,
    ) -> NumericalGuardResult:
        return self._require_validator().numerical_guard(proposal)

    def _commit_without_full_safety(
        self,
        proposal: ConsolidationProposal,
        requests: list[EditRequest],
        *,
        safety_status: str,
    ) -> bool:
        try:
            proposal.apply()
            proposal.commit()
        except Exception as error:
            proposal.rollback()
            self._enter_external_only("commit_error", error=error)
            return False
        self._finish_safe_consolidation(
            requests,
            None,
            safety_status=safety_status,
        )
        return True

    def _validate_and_finish(
        self,
        proposal: ConsolidationProposal,
        requests: list[EditRequest],
        numerical: NumericalGuardResult,
    ) -> bool:
        try:
            result = self._require_validator().validate_candidate(
                proposal,
                requests,
                self._require_writer(),
                consolidation_count=self._consolidation_count,
                numerical=numerical,
            )
        except BaseException as error:
            proposal.rollback()
            if not isinstance(error, Exception):
                raise
            self._enter_external_only("validation_error", error=error)
            return False

        if not result.safe:
            proposal.rollback()
            self._enter_external_only(
                "safety_veto",
                result=result,
            )
            return False

        try:
            proposal.commit()
        except Exception as error:
            self._enter_external_only("commit_error", error=error)
            return False
        self._finish_safe_consolidation(
            requests,
            result,
            safety_status="checked",
        )
        return True

    def _finish_safe_consolidation(
        self,
        requests: list[EditRequest],
        result: ConsolidationSafetyResult | None,
        *,
        safety_status: str,
    ) -> None:
        if self._validator is not None:
            self._validator.record_requests(requests)
        self.storage.clear()
        self._refresh_routing_thresholds()
        self._consolidation_count += 1
        self._consolidated_edit_count += len(requests)
        report = {
            "safe": True,
            "cycle": self._consolidation_count,
            "consolidated_edits": len(requests),
            "total_consolidated_edits": self._consolidated_edit_count,
            "external_only": False,
            "safety_status": safety_status,
        }
        if result is not None:
            report["safety"] = asdict(result)
            report["safety_summary"] = consolidation_safety_summary(
                result,
                self.config.safety_config,
            )
        self._record_consolidation_report(report)

    def _enter_external_only(
        self,
        reason: str,
        *,
        error: Exception | None = None,
        metrics: dict | None = None,
        result: ConsolidationSafetyResult | None = None,
    ) -> None:
        self._external_only = True
        self._refresh_routing_thresholds(force=True)
        if result is not None:
            safety_status = "checked"
        elif reason == "numerical":
            safety_status = "numerical"
        else:
            safety_status = "error"
        report = {
            "safe": False,
            "cycle": self._consolidation_count + 1,
            "retained_external_edits": self.external_memory_size,
            "total_consolidated_edits": self._consolidated_edit_count,
            "external_only": True,
            "safety_status": safety_status,
            "reasons": [reason],
        }
        if error is not None:
            report["error"] = f"{type(error).__name__}: {error}"
        if metrics is not None:
            report["metrics"] = metrics
        if result is not None:
            report["safety"] = asdict(result)
            report["safety_summary"] = consolidation_safety_summary(
                result,
                self.config.safety_config,
            )
            report["reasons"] = list(result.reasons) or [reason]
        self._record_consolidation_report(report)
        self._release_consolidation_state()

    def _record_consolidation_report(self, report: dict) -> None:
        self.last_consolidation_report = report
        self.consolidation_history.append(report)
        diagnostics = report.get("safety_summary", report.get("metrics", {}))
        if "error" in report:
            diagnostics = {**diagnostics, "error": report["error"]}
        self._pending_log_events.append({
            "event": "consolidation",
            "safe": report["safe"],
            "external_only": report["external_only"],
            "safety_status": report["safety_status"],
            "reasons": report.get("reasons", []),
            "diagnostics": diagnostics,
        })

    def _release_consolidation_state(self) -> None:
        self.storage.compact_for_external_only()
        if self._validator is not None:
            self._validator.release()
            self._validator = None
        if self.writer is not None:
            self.writer.release_consolidation_state()
            if self._consolidation_count == 0:
                self.writer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def copy_for_generation(self, target: "Mnemo") -> None:
        if not isinstance(target, Mnemo):
            raise TypeError("Mnemo generation state can only be copied to Mnemo.")

        # Mnemo's external edits are CPU-backed; only consolidation layers can
        # differ between the source and a freshly loaded generation model.
        if not self.config.external_only:
            source_params = dict(self._base_model.named_parameters())
            target_params = dict(target._base_model.named_parameters())
            rewrite_template = self.config.consolidation_config.rewrite_module_tmp
            for layer in self.config.consolidation_config.layers:
                name = f"{rewrite_template.format(layer)}.weight"
                if name not in source_params or name not in target_params:
                    raise KeyError(
                        f"Mnemo generation copy cannot find consolidated weight {name!r}."
                    )
                with torch.no_grad():
                    target_params[name].copy_(source_params[name])

        target.storage.load_state(self.storage.copy_state())
        target._refresh_routing_thresholds(force=True)

        # Generation only needs the interceptor, router, and storage. Do not
        # carry the writer, validator, or raw consolidation history into it.
        target._release_consolidation_state()
        target._value_trainer = None
        target._require_interceptor().disarm()
        target.model.eval()
