from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random

import torch

from ...config import MnemoConfig
from ...utils import nethook
from ...utils.devices import get_model_device, get_module_device
from ..editor import Editor
from ..editor_utils import EditRequest
from .consolidation import ConsolidationProposal, ConsolidationWriter
from .ext_mem.interceptor import Interceptor, PrefillRoutingModel
from .ext_mem.router import build_router
from .safety import (
    BehavioralDriftResult,
    CanaryLogitBatch,
    ConsolidationSafetyResult,
    NumericalGuardResult,
    SphereDriftResult,
    assess_consolidation_safety,
    check_numerical_guard,
    evaluate_behavioral_drift,
    evaluate_retention,
    evaluate_sphere_drift,
)
from .ext_mem.storage import MnemoStorage


class Mnemo(Editor):
    """Hybrid external memory with transactional weight consolidation."""

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
        self._retention_reservoir: list[EditRequest] = []
        self._retention_seen = 0
        self._reservoir_rng = random.Random(0)
        self._consolidation_count = 0
        self._consolidated_edit_count = 0
        self._external_only = config.external_only
        self._routing_profile_max_size = routing_profile.max_bank_size
        self._closed = True
        self._original_canary_logits: torch.Tensor | None = None
        self._canary_target_ids: torch.Tensor | None = None
        self.last_delta_loss: float | None = None
        self.last_delta_norm: float | None = None
        self.last_delta_saturated: bool | None = None
        self.last_consolidation_report: dict | None = None
        self.consolidation_history: list[dict] = []
        self._install_runtime()
        if self.config.safety_config.enabled and not self._external_only:
            # Preserve the pretrained reference before deferred consolidations begin.
            self._ensure_original_canaries()

    def edit(self, request: EditRequest) -> None:
        self._install_runtime()
        key = self.router.encode([request.prompt])
        value, delta_norm, loss = self._train_value(request)
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
            print(
                "[mnemo-edit] "
                + json.dumps(
                    {
                        "memory_index": memory_index,
                        "prompt": request.prompt,
                        "answer": request.answer,
                        "final_loss": self.last_delta_loss,
                        "delta_norm": self.last_delta_norm,
                        "max_norm": max_norm,
                        "max_norm_reached": self.last_delta_saturated,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )

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
        self._retention_reservoir.clear()
        self._retention_seen = 0
        self._reservoir_rng.seed(0)
        self._consolidation_count = 0
        self._consolidated_edit_count = 0
        self._external_only = self.config.external_only
        self._refresh_routing_thresholds(force=True)
        self._original_canary_logits = None
        self._canary_target_ids = None
        self.last_delta_loss = None
        self.last_delta_norm = None
        self.last_delta_saturated = None
        self.last_consolidation_report = None
        self.consolidation_history.clear()
        if self._interceptor is not None:
            self._interceptor.close()
            self._interceptor = None
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
        self._interceptor_or_raise().disarm()
        writer = self._writer_or_raise()
        try:
            proposal = writer.propose(requests)
        except Exception as error:
            self._enter_external_only("proposal_error", error=error)
            return False

        numerical = self._proposal_numerical_guard(proposal)
        if not numerical.safe:
            proposal.rollback()
            self._enter_external_only(
                "numerical",
                metrics={
                    "maximum_relative_update_norm": (
                        numerical.maximum_relative_update_norm
                    )
                },
            )
            return False

        safety_config = self.config.safety_config
        if not safety_config.enabled:
            safe = self._commit_without_full_safety(
                proposal,
                requests,
                safety_status="disabled",
            )
        elif self._consolidated_edit_count < safety_config.start_after_edits:
            try:
                self._ensure_original_canaries()
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
        self._freeze_base_model()
        self.tokenizer.padding_side = "right"
        block_name = self.config.intervention_module_tmp.format(
            self.config.intervention_block
        )
        block = nethook.get_module(self._base_model, block_name)
        self._interceptor = Interceptor(block)
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

    def _interceptor_or_raise(self) -> Interceptor:
        if self._interceptor is None:
            raise RuntimeError("Mnemo interceptor is not installed.")
        return self._interceptor

    def _writer_or_raise(self) -> ConsolidationWriter:
        if self.writer is None:
            raise RuntimeError("External-only Mnemo has no consolidation writer.")
        return self.writer

    def _train_value(
        self,
        request: EditRequest,
    ) -> tuple[torch.Tensor, float, float]:
        prompt_ids, target_ids = self._request_token_ids(request)
        input_ids = torch.tensor(
            [prompt_ids + target_ids],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[:, len(prompt_ids):] = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=self.device,
        )

        baseline = self._intervention_state(
            input_ids,
            attention_mask,
            len(prompt_ids) - 1,
        ).float()
        hidden_size = self._hidden_size()
        delta = torch.nn.Parameter(
            torch.zeros((1, hidden_size), dtype=torch.float32, device=self.device)
        )
        delta_config = self.config.delta_config
        optimizer = torch.optim.Adam(
            [delta],
            lr=delta_config.learning_rate,
            weight_decay=delta_config.weight_decay,
            foreach=False,
        )
        was_training = self._base_model.training
        self._base_model.eval()
        final_loss = float("inf")
        try:
            with self._interceptor_or_raise().arm_training_delta(
                delta,
                len(prompt_ids) - 1,
            ):
                for _ in range(delta_config.num_steps):
                    optimizer.zero_grad(set_to_none=True)
                    outputs = self._base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = outputs.loss.float()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Mnemo delta loss became non-finite.")
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        norm = torch.linalg.vector_norm(delta)
                        if norm > delta_config.max_norm:
                            delta.mul_(delta_config.max_norm / norm)
                    final_loss = float(loss.detach().cpu())
                    del outputs, loss
                    if final_loss <= delta_config.early_stop_loss:
                        break
        finally:
            self._base_model.train(was_training)
            del optimizer, input_ids, attention_mask, labels

        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Mnemo delta became non-finite.")
        delta_norm = float(torch.linalg.vector_norm(delta.detach().float()))
        block_dtype = next(
            nethook.get_module(
                self._base_model,
                self.config.intervention_module_tmp.format(
                    self.config.intervention_block
                ),
            ).parameters()
        ).dtype
        value = baseline + delta.detach()
        return (
            value.to(device="cpu", dtype=block_dtype),
            delta_norm,
            final_loss,
        )

    def _intervention_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_index: int,
    ) -> torch.Tensor:
        block = nethook.get_module(
            self._base_model,
            self.config.intervention_module_tmp.format(
                self.config.intervention_block
            ),
        )
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output) -> None:
            hidden = nethook.get_hidden_state(output)
            captured.append(hidden[:, token_index, :].detach())

        handle = block.register_forward_hook(capture)
        was_training = self._base_model.training
        self._base_model.eval()
        try:
            with torch.inference_mode():
                self._base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
            self._base_model.train(was_training)
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one intervention state, captured {len(captured)}."
            )
        return captured[0]

    def _request_token_ids(
        self,
        request: EditRequest,
    ) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(request.prompt, add_special_tokens=True)
        target = request.answer
        if not target:
            raise ValueError("Mnemo edit targets must not be empty.")
        if (
            self.config.delta_config.target_leading_space
            and request.prompt
            and not request.prompt[-1].isspace()
            and not target[0].isspace()
        ):
            target = f" {target}"
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not prompt_ids or not target_ids:
            raise ValueError("Mnemo prompt and target must produce at least one token.")
        return list(prompt_ids), list(target_ids)

    def _hidden_size(self) -> int:
        model_config = getattr(self._base_model.config, "text_config", None)
        model_config = model_config or self._base_model.config
        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("The language model config has no hidden_size.")
        return int(hidden_size)

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
        current_weights = dict(self._base_model.named_parameters())
        return check_numerical_guard(
            proposal.deltas,
            {name: current_weights[name] for name in proposal.deltas},
            self.config.safety_config,
        )

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
        safety_config = self.config.safety_config
        try:
            self._ensure_original_canaries()
            last_safe_logits = self._canary_logits()
            retained = self._sample_retained_requests()
            retained_reference = self._target_token_scores(retained) if retained else [1.0]

            proposal.apply()
            candidate_logits = self._canary_logits()
            current_scores = self._target_token_scores(requests)
            retained_scores = self._target_token_scores(retained) if retained else [1.0]

            behavioral = self._behavioral_result(
                candidate_logits,
                last_safe_logits,
            )
            retention = evaluate_retention(
                current_scores,
                retained_scores,
                safety_config,
                retained_reference,
            )
            sphere = self._sphere_result(proposal)
            result = assess_consolidation_safety(
                numerical,
                behavioral,
                retention,
                sphere,
                safety_config,
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

    def _ensure_original_canaries(self) -> None:
        if self._original_canary_logits is not None:
            return
        prompts = self.config.safety_config.canary_prompts[
            :self.config.safety_config.canary_sample_size
        ]
        logits = self._canary_logits(prompts)
        self._original_canary_logits = logits
        self._canary_target_ids = logits.argmax(dim=-1)

    def _canary_logits(self, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts is None:
            prompts = self.config.safety_config.canary_prompts[
                :self.config.safety_config.canary_sample_size
            ]
        microbatch = self.config.safety_config.validation_microbatch_size
        chunks = []
        for start in range(0, len(prompts), microbatch):
            batch = prompts[start:start + microbatch]
            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._transformer_hidden(
                    tokens["input_ids"],
                    tokens["attention_mask"],
                )
                positions = torch.stack([
                    torch.nonzero(mask, as_tuple=False)[-1, 0]
                    for mask in tokens["attention_mask"]
                ])
                rows = torch.arange(len(batch), device=hidden.device)
                selected = hidden[rows, positions.to(hidden.device)]
                logits = self._lm_head(selected).float().cpu()
            chunks.append(logits)
            del tokens, hidden, selected, logits
        return torch.cat(chunks, dim=0)

    def _target_token_scores(self, requests: list[EditRequest]) -> list[float]:
        if not requests:
            return []
        microbatch = self.config.safety_config.validation_microbatch_size
        scores: list[float] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Mnemo safety validation requires a tokenizer pad token.")

        for start in range(0, len(requests), microbatch):
            batch = requests[start:start + microbatch]
            tokenized = [self._request_token_ids(request) for request in batch]
            sequences = [prompt + target for prompt, target in tokenized]
            max_length = max(map(len, sequences))
            input_ids = torch.full(
                (len(batch), max_length),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, sequence in enumerate(sequences):
                input_ids[row, :len(sequence)] = torch.tensor(
                    sequence,
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask[row, :len(sequence)] = 1

            with torch.inference_mode():
                hidden = self._transformer_hidden(input_ids, attention_mask)
                selected_rows = []
                expected_ids = []
                owners = []
                for row, (prompt_ids, target_ids) in enumerate(tokenized):
                    positions = torch.arange(
                        len(prompt_ids) - 1,
                        len(prompt_ids) + len(target_ids) - 1,
                        device=hidden.device,
                    )
                    selected_rows.append(hidden[row, positions])
                    expected_ids.extend(target_ids)
                    owners.extend([row] * len(target_ids))
                selected = torch.cat(selected_rows, dim=0)
                predictions = self._lm_head(selected).argmax(dim=-1).cpu()
            expected = torch.tensor(expected_ids, dtype=torch.long)
            owner_tensor = torch.tensor(owners, dtype=torch.long)
            correct = predictions.eq(expected)
            for row in range(len(batch)):
                scores.append(float(correct[owner_tensor == row].float().mean()))
            del input_ids, attention_mask, hidden, selected, predictions
        return scores

    def _transformer_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        transformer = getattr(self._base_model, "model", None)
        if transformer is None:
            raise AttributeError("Mnemo expects a causal LM with a .model transformer.")
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        head_path = self.config.consolidation_config.lm_head_module
        head = nethook.get_module(self._base_model, head_path)
        head_device = get_module_device(
            self._base_model,
            head_path,
            fallback=hidden.device,
        )
        return head(hidden.to(head_device))

    def _behavioral_result(
        self,
        candidate_logits: torch.Tensor,
        last_safe_logits: torch.Tensor,
    ) -> BehavioralDriftResult:
        if self._original_canary_logits is None or self._canary_target_ids is None:
            raise RuntimeError("Original canary state is not initialized.")
        microbatch = self.config.safety_config.validation_microbatch_size
        batches = []
        for start in range(0, len(candidate_logits), microbatch):
            stop = start + microbatch
            batches.append(CanaryLogitBatch(
                candidate_logits=candidate_logits[start:stop],
                last_safe_logits=last_safe_logits[start:stop],
                original_logits=self._original_canary_logits[start:stop],
                target_token_ids=self._canary_target_ids[start:stop],
            ))
        return evaluate_behavioral_drift(batches, self.config.safety_config)

    def _sphere_result(self, proposal: ConsolidationProposal) -> SphereDriftResult:
        params = dict(self._base_model.named_parameters())
        writer = self._writer_or_raise()
        results = [
            evaluate_sphere_drift(
                params[name],
                proposal.snapshots[name],
                writer.origin_weights[name],
                self.config.safety_config,
            )
            for name in proposal.deltas
        ]
        return SphereDriftResult(
            safe=all(result.safe for result in results),
            warning=any(result.warning for result in results),
            sample_count=sum(result.sample_count for result in results),
            candidate_energy=sum(result.candidate_energy for result in results) / len(results),
            last_safe_energy=sum(result.last_safe_energy for result in results) / len(results),
            original_energy=sum(result.original_energy for result in results) / len(results),
            step_relative_drift=max(result.step_relative_drift for result in results),
            cumulative_relative_drift=max(
                result.cumulative_relative_drift for result in results
            ),
        )

    def _sample_retained_requests(self) -> list[EditRequest]:
        reservoir = self._retention_reservoir
        count = min(len(reservoir), self.config.safety_config.retention_sample_size)
        if count == len(reservoir):
            return list(reservoir)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._consolidation_count)
        indices = torch.randperm(len(reservoir), generator=generator)[:count]
        return [reservoir[int(index)] for index in indices]

    def _finish_safe_consolidation(
        self,
        requests: list[EditRequest],
        result: ConsolidationSafetyResult | None,
        *,
        safety_status: str,
    ) -> None:
        for request in requests:
            self._add_retention_request(request)
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
        self.last_consolidation_report = report
        self.consolidation_history.append(report)

    def _add_retention_request(self, request: EditRequest) -> None:
        self._retention_seen += 1
        capacity = self.config.safety_config.retention_reservoir_size
        if len(self._retention_reservoir) < capacity:
            self._retention_reservoir.append(request)
            return
        replacement = self._reservoir_rng.randrange(self._retention_seen)
        if replacement < capacity:
            self._retention_reservoir[replacement] = request

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
        report = {
            "safe": False,
            "cycle": self._consolidation_count + 1,
            "retained_external_edits": self.external_memory_size,
            "total_consolidated_edits": self._consolidated_edit_count,
            "external_only": True,
            "reasons": [reason],
        }
        if error is not None:
            report["error"] = f"{type(error).__name__}: {error}"
        if metrics is not None:
            report["metrics"] = metrics
        if result is not None:
            report["safety"] = asdict(result)
            report["reasons"] = list(result.reasons) or [reason]
        self.last_consolidation_report = report
        self.consolidation_history.append(report)
        self._release_hybrid_only_state()

    def _release_hybrid_only_state(self) -> None:
        self.storage.compact_for_external_only()
        self._retention_reservoir.clear()
        self._retention_seen = 0
        self._original_canary_logits = None
        self._canary_target_ids = None
        if self.writer is not None:
            self.writer.release_consolidation_state()
            if self._consolidation_count == 0:
                self.writer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sync_generation_state_to(self, target: "Mnemo") -> None:
        if not isinstance(target, Mnemo):
            raise TypeError("Mnemo generation state can only be copied to Mnemo.")
        target._base_model.load_state_dict(self._base_model.state_dict(), strict=True)
        target.storage.load_state(self.storage.copy_state())
        target._external_only = self._external_only
        target._refresh_routing_thresholds(force=True)
        if target._external_only:
            # Generation-only copies must not retain Mnemo's consolidation snapshots
            # or safety reservoir after the source has switched to external memory.
            target._release_hybrid_only_state()
        target._interceptor_or_raise().disarm()
        target.model.eval()
