import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....config.mnemo.mnemo_config import MnemoDeltaConfig, MnemoSafetyConfig
from ....utils import nethook
from ....utils.devices import get_module_device
from ...editor_utils import EditRequest
from ..ext_mem.value import request_token_ids
from ..safety import (
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
from .writer import ConsolidationProposal, ConsolidationWriter


class ConsolidationValidator:
    """Owns Mnemo's stateful pre-commit consolidation checks."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        safety_config: MnemoSafetyConfig,
        delta_config: MnemoDeltaConfig,
        *,
        device: torch.device | str,
        lm_head_module: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.safety_config = safety_config
        self.delta_config = delta_config
        self.device = device
        self.lm_head_module = lm_head_module
        self._retention_reservoir: list[EditRequest] = []
        self._retention_seen = 0
        self._reservoir_rng = random.Random(0)
        self._original_canary_logits: torch.Tensor | None = None
        self._canary_target_ids: torch.Tensor | None = None

    def reset(self) -> None:
        self._retention_reservoir.clear()
        self._retention_seen = 0
        self._reservoir_rng.seed(0)
        self._original_canary_logits = None
        self._canary_target_ids = None

    def release(self) -> None:
        self._retention_reservoir.clear()
        self._retention_seen = 0
        self._original_canary_logits = None
        self._canary_target_ids = None

    def ensure_reference(self) -> None:
        if self._original_canary_logits is not None:
            return
        prompts = self.safety_config.canary_prompts[
            :self.safety_config.canary_sample_size
        ]
        logits = self._canary_logits(prompts)
        self._original_canary_logits = logits
        self._canary_target_ids = logits.argmax(dim=-1)

    def numerical_guard(
        self,
        proposal: ConsolidationProposal,
    ) -> NumericalGuardResult:
        current_weights = dict(self.model.named_parameters())
        return check_numerical_guard(
            proposal.deltas,
            {name: current_weights[name] for name in proposal.deltas},
            self.safety_config,
        )

    def validate_candidate(
        self,
        proposal: ConsolidationProposal,
        requests: list[EditRequest],
        writer: ConsolidationWriter,
        *,
        consolidation_count: int,
        numerical: NumericalGuardResult,
    ) -> ConsolidationSafetyResult:
        self.ensure_reference()
        last_safe_logits = self._canary_logits()
        retained = self._sample_retained_requests(consolidation_count)
        retained_reference = self._target_token_scores(retained) if retained else [1.0]

        proposal.apply()
        candidate_logits = self._canary_logits()
        current_scores = self._target_token_scores(requests)
        retained_scores = self._target_token_scores(retained) if retained else [1.0]

        behavioral = self._behavioral_result(candidate_logits, last_safe_logits)
        retention = evaluate_retention(
            current_scores,
            retained_scores,
            self.safety_config,
            retained_reference,
        )
        sphere = self._sphere_result(proposal, writer)
        return assess_consolidation_safety(
            numerical,
            behavioral,
            retention,
            sphere,
            self.safety_config,
        )

    def record_requests(self, requests: list[EditRequest]) -> None:
        for request in requests:
            self._add_retention_request(request)

    def _canary_logits(self, prompts: list[str] | None = None) -> torch.Tensor:
        if prompts is None:
            prompts = self.safety_config.canary_prompts[
                :self.safety_config.canary_sample_size
            ]
        microbatch = self.safety_config.validation_microbatch_size
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
        microbatch = self.safety_config.validation_microbatch_size
        scores: list[float] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Mnemo safety validation requires a tokenizer pad token.")

        for start in range(0, len(requests), microbatch):
            batch = requests[start:start + microbatch]
            tokenized = [
                request_token_ids(self.tokenizer, request, self.delta_config)
                for request in batch
            ]
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
        transformer = getattr(self.model, "model", None)
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
        head = nethook.get_module(self.model, self.lm_head_module)
        head_device = get_module_device(
            self.model,
            self.lm_head_module,
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
        microbatch = self.safety_config.validation_microbatch_size
        batches = []
        for start in range(0, len(candidate_logits), microbatch):
            stop = start + microbatch
            batches.append(CanaryLogitBatch(
                candidate_logits=candidate_logits[start:stop],
                last_safe_logits=last_safe_logits[start:stop],
                original_logits=self._original_canary_logits[start:stop],
                target_token_ids=self._canary_target_ids[start:stop],
            ))
        return evaluate_behavioral_drift(batches, self.safety_config)

    def _sphere_result(
        self,
        proposal: ConsolidationProposal,
        writer: ConsolidationWriter,
    ) -> SphereDriftResult:
        params = dict(self.model.named_parameters())
        results = [
            evaluate_sphere_drift(
                params[name],
                proposal.snapshots[name],
                writer.origin_weights[name],
                self.safety_config,
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

    def _sample_retained_requests(
        self,
        consolidation_count: int,
    ) -> list[EditRequest]:
        reservoir = self._retention_reservoir
        count = min(len(reservoir), self.safety_config.retention_sample_size)
        if count == len(reservoir):
            return list(reservoir)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(consolidation_count)
        indices = torch.randperm(len(reservoir), generator=generator)[:count]
        return [reservoir[int(index)] for index in indices]

    def _add_retention_request(self, request: EditRequest) -> None:
        self._retention_seen += 1
        capacity = self.safety_config.retention_reservoir_size
        if len(self._retention_reservoir) < capacity:
            self._retention_reservoir.append(request)
            return
        replacement = self._reservoir_rng.randrange(self._retention_seen)
        if replacement < capacity:
            self._retention_reservoir[replacement] = request
