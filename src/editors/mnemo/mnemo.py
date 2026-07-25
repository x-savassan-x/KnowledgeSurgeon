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
from .consolidation import AlphaEditProposal, AlphaEditWriter
from .interceptor import DecoderBlockDeltaInterceptor, PrefillRoutingModel
from .router import BgeM3Router, ExternalMemoryBank
from .safety import (
    BehavioralDriftResult,
    CanaryLogitBatch,
    ConsolidationSafetyResult,
    SphereDriftResult,
    assess_consolidation_safety,
    check_numerical_guard,
    evaluate_behavioral_drift,
    evaluate_retention,
    evaluate_sphere_drift,
)


class Mnemo(Editor):
    """Hybrid external memory with transactional AlphaEdit consolidation."""

    config: MnemoConfig

    def __init__(self, config: MnemoConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self._base_model = self.model
        self.device = get_model_device(self._base_model, fallback=config.device)
        self.config.device = self.device
        if not config.external_only:
            projector_path = Path(config.P_loc)
            if not projector_path.exists():
                raise FileNotFoundError(
                    f"MNEMO requires the AlphaEdit projector: {projector_path}. "
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
        self.router = BgeM3Router(
            score_threshold=router_config.score_threshold,
            margin_threshold=router_config.margin_threshold,
            model_name=router_config.model_name,
            device=router_config.device,
            use_fp16=router_config.fp16,
            max_length=router_config.max_length,
            batch_size=router_config.batch_size,
        )
        self.memory = ExternalMemoryBank(
            score_threshold=router_config.score_threshold,
            margin_threshold=router_config.margin_threshold,
        )
        self.writer = (
            None
            if config.external_only
            else AlphaEditWriter(self._base_model, self.tokenizer, config)
        )

        self._interceptor: DecoderBlockDeltaInterceptor | None = None
        self._pending_requests: list[EditRequest] = []
        self._retention_reservoir: list[EditRequest] = []
        self._retention_seen = 0
        self._reservoir_rng = random.Random(0)
        self._consolidation_count = 0
        self._external_only = config.external_only
        self._closed = True
        self._original_canary_logits: torch.Tensor | None = None
        self._canary_target_ids: torch.Tensor | None = None
        self.last_delta_loss: float | None = None
        self.last_delta_norm: float | None = None
        self.last_delta_saturated: bool | None = None
        self.last_consolidation_report: dict | None = None
        self.consolidation_history: list[dict] = []
        self._install_runtime()

    @property
    def buffer_size(self) -> int:
        return len(self._pending_requests)

    @property
    def external_only(self) -> bool:
        return self._external_only

    @property
    def external_memory_size(self) -> int:
        return len(self.memory)

    def edit(self, request: EditRequest) -> None:
        self._install_runtime()
        key = self.router.encode([request.prompt])
        delta, loss = self._train_delta(request)
        memory_index = self.memory.add(key, delta, metadata=request)
        if not self._external_only:
            self._pending_requests.append(request)
        self.last_delta_loss = loss
        self.last_delta_norm = float(torch.linalg.vector_norm(delta.float()))
        max_norm = self.config.delta_config.max_norm
        self.last_delta_saturated = self.last_delta_norm >= max_norm * (1.0 - 1.0e-4)
        if self.config.diagnostics:
            print(
                "[mnemo-edit] "
                + json.dumps(
                    {
                        "memory_index": memory_index,
                        "prompt": request.prompt,
                        "target": request.new_ground_truth,
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

    def consolidate(self) -> bool:
        if not self._pending_requests or self._external_only:
            return False

        requests = list(self._pending_requests)
        self._interceptor_or_raise().disarm()
        writer = self._writer_or_raise()
        try:
            proposal = writer.propose(requests)
        except Exception as error:
            self._enter_external_only("proposal_error", error=error)
            return False

        if not self.config.safety_config.enabled:
            try:
                proposal.apply()
                proposal.commit()
            except Exception as error:
                proposal.rollback()
                self._enter_external_only("commit_error", error=error)
                return False
            self._finish_safe_consolidation(requests, None)
            return True

        safe = self._validate_and_finish(proposal, requests)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return safe

    def reset(self) -> None:
        if self.writer is not None:
            self.writer.reset()
        self.memory.clear()
        self.router.clear()
        self._pending_requests.clear()
        self._retention_reservoir.clear()
        self._retention_seen = 0
        self._reservoir_rng.seed(0)
        self._consolidation_count = 0
        self._external_only = self.config.external_only
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

    def sync_generation_state_to(self, target: "Mnemo") -> None:
        if not isinstance(target, Mnemo):
            raise TypeError("MNEMO generation state can only be copied to MNEMO.")
        target._base_model.load_state_dict(self._base_model.state_dict(), strict=True)
        target.memory.load_state(self.memory.copy_state())
        target._external_only = self._external_only
        target._interceptor_or_raise().disarm()
        target.model.eval()

    def _install_runtime(self) -> None:
        if not self._closed:
            return
        if self.writer is None and not self.config.external_only:
            projector_path = Path(self.config.P_loc)
            if not projector_path.exists():
                raise FileNotFoundError(
                    f"MNEMO requires the AlphaEdit projector: {projector_path}. "
                    "Download it with notebook/assets.ipynb before editing."
                )
            self.writer = AlphaEditWriter(
                self._base_model,
                self.tokenizer,
                self.config,
            )
        self._freeze_base_model()
        self.tokenizer.padding_side = "right"
        block_name = self.config.intervention_module_tmp.format(
            self.config.intervention_block
        )
        block = nethook.get_module(self._base_model, block_name)
        self._interceptor = DecoderBlockDeltaInterceptor(block)
        self.model = PrefillRoutingModel(
            self._base_model,
            self.tokenizer,
            self.router,
            self._interceptor,
            memory=self.memory,
            diagnostics=self.config.diagnostics,
        )
        self._closed = False

    def _freeze_base_model(self) -> None:
        for parameter in self._base_model.parameters():
            parameter.requires_grad_(False)

    def _interceptor_or_raise(self) -> DecoderBlockDeltaInterceptor:
        if self._interceptor is None:
            raise RuntimeError("MNEMO interceptor is not installed.")
        return self._interceptor

    def _writer_or_raise(self) -> AlphaEditWriter:
        if self.writer is None:
            raise RuntimeError("External-only MNEMO has no consolidation writer.")
        return self.writer

    def _train_delta(self, request: EditRequest) -> tuple[torch.Tensor, float]:
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
            with self._interceptor_or_raise().arm_training(
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
                        raise FloatingPointError("MNEMO delta loss became non-finite.")
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
            raise FloatingPointError("MNEMO delta became non-finite.")
        block_dtype = next(
            nethook.get_module(
                self._base_model,
                self.config.intervention_module_tmp.format(
                    self.config.intervention_block
                ),
            ).parameters()
        ).dtype
        return delta.detach().to(device="cpu", dtype=block_dtype), final_loss

    def _request_token_ids(self, request: EditRequest) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(request.prompt, add_special_tokens=True)
        target = request.new_ground_truth
        if not target:
            raise ValueError("MNEMO edit targets must not be empty.")
        if (
            self.config.delta_config.target_leading_space
            and request.prompt
            and not request.prompt[-1].isspace()
            and not target[0].isspace()
        ):
            target = f" {target}"
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        if not prompt_ids or not target_ids:
            raise ValueError("MNEMO prompt and target must produce at least one token.")
        return list(prompt_ids), list(target_ids)

    def _hidden_size(self) -> int:
        model_config = getattr(self._base_model.config, "text_config", None)
        model_config = model_config or self._base_model.config
        hidden_size = getattr(model_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("The language model config has no hidden_size.")
        return int(hidden_size)

    def _validate_and_finish(
        self,
        proposal: AlphaEditProposal,
        requests: list[EditRequest],
    ) -> bool:
        safety_config = self.config.safety_config
        current_weights = dict(self._base_model.named_parameters())
        numerical = check_numerical_guard(
            proposal.deltas,
            {name: current_weights[name] for name in proposal.deltas},
            safety_config,
        )
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
        self._finish_safe_consolidation(requests, result)
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
            raise ValueError("MNEMO safety validation requires a tokenizer pad token.")

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
            raise AttributeError("MNEMO expects a causal LM with a .model transformer.")
        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        head = nethook.get_module(self._base_model, self.config.lm_head_module)
        head_device = get_module_device(
            self._base_model,
            self.config.lm_head_module,
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

    def _sphere_result(self, proposal: AlphaEditProposal) -> SphereDriftResult:
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
    ) -> None:
        for request in requests:
            self._add_retention_request(request)
        self._pending_requests.clear()
        self.memory.clear()
        self._consolidation_count += 1
        report = {
            "safe": True,
            "cycle": self._consolidation_count,
            "consolidated_edits": len(requests),
            "external_only": False,
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
        report = {
            "safe": False,
            "cycle": self._consolidation_count + 1,
            "retained_external_edits": self.external_memory_size,
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
        self._pending_requests.clear()
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
