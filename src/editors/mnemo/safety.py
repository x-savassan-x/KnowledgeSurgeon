from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

import torch
import torch.nn.functional as F

from ...config.mnemo.mnemo_config import MnemoSafetyConfig


@dataclass(frozen=True)
class CanaryLogitBatch:
    candidate_logits: torch.Tensor
    last_safe_logits: torch.Tensor
    original_logits: torch.Tensor
    target_token_ids: torch.Tensor


@dataclass(frozen=True)
class NumericalGuardResult:
    safe: bool
    warning: bool
    nonfinite_tensors: tuple[str, ...]
    relative_update_norms: dict[str, float]
    maximum_relative_update_norm: float


@dataclass(frozen=True)
class BehavioralDriftResult:
    safe: bool
    warning: bool
    sample_count: int
    step_kl: float
    cumulative_kl: float
    step_nll_change: float
    cumulative_nll_change: float


@dataclass(frozen=True)
class RetentionResult:
    safe: bool
    warning: bool
    current_batch_success: float
    retained_success: float
    retained_reference_success: float | None
    retained_mean_drop: float | None


@dataclass(frozen=True)
class SphereDriftResult:
    safe: bool
    warning: bool
    sample_count: int
    candidate_energy: float
    last_safe_energy: float
    original_energy: float
    step_relative_drift: float
    cumulative_relative_drift: float


@dataclass(frozen=True)
class ConsolidationSafetyResult:
    safe: bool
    reasons: tuple[str, ...]
    warning_count: int
    numerical: NumericalGuardResult
    behavioral: BehavioralDriftResult
    retention: RetentionResult
    sphere: SphereDriftResult
    metrics: dict[str, float | int | bool] = field(default_factory=dict)


def deterministic_sample_indices(
    length: int,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    if length < 0:
        raise ValueError("length must be nonnegative.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    count = min(length, sample_size)
    if count == length:
        return torch.arange(length, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(length, generator=generator)[:count].sort().values


def _tensor_chunks(tensor: torch.Tensor, chunk_elements: int):
    value = tensor.detach()
    if value.is_contiguous() or value.ndim <= 1:
        flat = value.reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            yield flat[start : start + chunk_elements]
        return

    elements_per_row = value[0].numel()
    rows_per_chunk = max(1, chunk_elements // max(1, elements_per_row))
    for start in range(0, len(value), rows_per_chunk):
        yield value[start : start + rows_per_chunk].contiguous().view(-1)


def _is_finite_chunked(tensor: torch.Tensor, chunk_elements: int) -> bool:
    return all(
        bool(torch.isfinite(chunk).all().item())
        for chunk in _tensor_chunks(tensor, chunk_elements)
    )


def _squared_l2_chunked(tensor: torch.Tensor, chunk_elements: int) -> float:
    total = 0.0
    for chunk in _tensor_chunks(tensor, chunk_elements):
        chunk_float = chunk.float()
        total += float(torch.dot(chunk_float, chunk_float).item())
    return total


def check_numerical_guard(
    updates: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    config: MnemoSafetyConfig,
) -> NumericalGuardResult:
    missing = sorted(set(updates) - set(weights))
    if missing:
        raise KeyError(f"Missing reference weights for updates: {missing}")

    nonfinite = []
    relative_norms = {}
    for name, update in updates.items():
        weight = weights[name]
        if update.shape != weight.shape:
            raise ValueError(
                f"Update shape for {name} is {tuple(update.shape)}, "
                f"expected {tuple(weight.shape)}."
            )
        update_finite = _is_finite_chunked(update, config.tensor_chunk_elements)
        weight_finite = _is_finite_chunked(weight, config.tensor_chunk_elements)
        if not update_finite:
            nonfinite.append(f"update:{name}")
        if not weight_finite:
            nonfinite.append(f"weight:{name}")
        if not update_finite or not weight_finite:
            relative_norms[name] = float("inf")
            continue
        update_norm = _squared_l2_chunked(
            update, config.tensor_chunk_elements
        ) ** 0.5
        weight_norm = _squared_l2_chunked(
            weight, config.tensor_chunk_elements
        ) ** 0.5
        relative_norms[name] = update_norm / max(weight_norm, 1e-12)

    maximum = max(relative_norms.values(), default=0.0)
    safe = not nonfinite and maximum < config.relative_update_norm_hard_limit
    warning = bool(nonfinite) or maximum >= config.relative_update_norm_warning
    return NumericalGuardResult(
        safe=safe,
        warning=warning,
        nonfinite_tensors=tuple(nonfinite),
        relative_update_norms=relative_norms,
        maximum_relative_update_norm=maximum,
    )


def _validate_canary_batch(batch: CanaryLogitBatch) -> None:
    shapes = {
        tuple(batch.candidate_logits.shape),
        tuple(batch.last_safe_logits.shape),
        tuple(batch.original_logits.shape),
    }
    if len(shapes) != 1:
        raise ValueError("Candidate, last-safe, and original logits must match.")
    if batch.candidate_logits.ndim != 2:
        raise ValueError("Canary logits must have shape [batch, vocabulary].")
    if batch.target_token_ids.ndim != 1:
        raise ValueError("target_token_ids must have shape [batch].")
    if len(batch.target_token_ids) != len(batch.candidate_logits):
        raise ValueError("Every canary row must have one target token id.")


def evaluate_behavioral_drift(
    batches: Iterable[CanaryLogitBatch],
    config: MnemoSafetyConfig,
) -> BehavioralDriftResult:
    count = 0
    step_kl_total = 0.0
    cumulative_kl_total = 0.0
    step_nll_total = 0.0
    cumulative_nll_total = 0.0

    for batch in batches:
        _validate_canary_batch(batch)
        candidate = F.log_softmax(batch.candidate_logits.float(), dim=-1)
        last_safe = F.log_softmax(
            batch.last_safe_logits.to(candidate.device, dtype=torch.float32),
            dim=-1,
        )
        original = F.log_softmax(
            batch.original_logits.to(candidate.device, dtype=torch.float32),
            dim=-1,
        )
        target_ids = batch.target_token_ids.to(candidate.device, dtype=torch.long)
        rows = torch.arange(len(candidate), device=candidate.device)

        step_kl_total += float(
            F.kl_div(candidate, last_safe, reduction="sum", log_target=True).item()
        )
        cumulative_kl_total += float(
            F.kl_div(candidate, original, reduction="sum", log_target=True).item()
        )
        step_nll_total += float(
            (last_safe[rows, target_ids] - candidate[rows, target_ids]).sum().item()
        )
        cumulative_nll_total += float(
            (original[rows, target_ids] - candidate[rows, target_ids]).sum().item()
        )
        count += len(candidate)

    if count == 0:
        raise ValueError("Behavioral drift requires at least one canary row.")

    step_kl = step_kl_total / count
    cumulative_kl = cumulative_kl_total / count
    step_nll = step_nll_total / count
    cumulative_nll = cumulative_nll_total / count
    values = (step_kl, cumulative_kl, step_nll, cumulative_nll)
    safe = all(isfinite(value) for value in values) and not any((
        step_kl >= config.step_kl_hard_limit,
        cumulative_kl >= config.cumulative_kl_hard_limit,
        step_nll >= config.step_nll_hard_limit,
        cumulative_nll >= config.cumulative_nll_hard_limit,
    ))
    warning = not all(isfinite(value) for value in values) or any((
        step_kl >= config.step_kl_warning,
        cumulative_kl >= config.cumulative_kl_warning,
        step_nll >= config.step_nll_warning,
        cumulative_nll >= config.cumulative_nll_warning,
    ))
    return BehavioralDriftResult(
        safe=safe,
        warning=warning,
        sample_count=count,
        step_kl=step_kl,
        cumulative_kl=cumulative_kl,
        step_nll_change=step_nll,
        cumulative_nll_change=cumulative_nll,
    )


def _mean_score(values: Sequence[float] | torch.Tensor, name: str) -> float:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor).all().item()):
        return float("nan")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any().item()):
        raise ValueError(f"{name} must contain success scores in [0, 1].")
    return float(tensor.mean().item())


def evaluate_retention(
    current_batch_scores: Sequence[float] | torch.Tensor,
    retained_scores: Sequence[float] | torch.Tensor,
    config: MnemoSafetyConfig,
    retained_reference_scores: Sequence[float] | torch.Tensor | None = None,
) -> RetentionResult:
    current_success = _mean_score(current_batch_scores, "current_batch_scores")
    retained_success = _mean_score(retained_scores, "retained_scores")
    reference_success = None
    mean_drop = None
    if retained_reference_scores is not None:
        reference_success = _mean_score(
            retained_reference_scores, "retained_reference_scores"
        )
        mean_drop = reference_success - retained_success

    values = [current_success, retained_success]
    if reference_success is not None:
        values.append(reference_success)
    finite = all(isfinite(value) for value in values)
    safe = finite and current_success >= config.current_batch_min_success
    safe = safe and retained_success >= config.retention_min_success
    if mean_drop is not None:
        safe = safe and mean_drop <= config.retention_max_mean_drop
    warning = not finite or not safe
    return RetentionResult(
        safe=safe,
        warning=warning,
        current_batch_success=current_success,
        retained_success=retained_success,
        retained_reference_success=reference_success,
        retained_mean_drop=mean_drop,
    )


def _hyperspherical_energy_for_indices(
    activations: torch.Tensor,
    indices: torch.Tensor,
    chunk_size: int,
    exponent: float,
) -> float:
    if activations.ndim != 2:
        raise ValueError("SPHERE activations must have shape [samples, features].")
    if len(indices) < 2:
        raise ValueError("SPHERE energy requires at least two sampled activations.")

    sampled = activations.detach().index_select(
        0, indices.to(activations.device)
    ).float()
    sampled = F.normalize(sampled, p=2, dim=1, eps=1e-12)
    total = 0.0
    pair_count = 0
    for start in range(0, len(sampled), chunk_size):
        stop = min(start + chunk_size, len(sampled))
        cosine = sampled[start:stop] @ sampled.T
        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * cosine, min=1e-12))
        row_indices = torch.arange(start, stop, device=sampled.device)
        column_indices = torch.arange(len(sampled), device=sampled.device)
        mask = column_indices.unsqueeze(0) > row_indices.unsqueeze(1)
        values = distances[mask].pow(-exponent)
        total += float(values.sum().item())
        pair_count += values.numel()
    return total / pair_count


def hyperspherical_energy(
    activations: torch.Tensor,
    sample_size: int,
    chunk_size: int,
    seed: int = 0,
    exponent: float = 1.0,
) -> float:
    indices = deterministic_sample_indices(len(activations), sample_size, seed)
    return _hyperspherical_energy_for_indices(
        activations, indices, chunk_size, exponent
    )


def evaluate_sphere_drift(
    candidate_activations: torch.Tensor,
    last_safe_activations: torch.Tensor,
    original_activations: torch.Tensor,
    config: MnemoSafetyConfig,
) -> SphereDriftResult:
    if not (
        len(candidate_activations)
        == len(last_safe_activations)
        == len(original_activations)
    ):
        raise ValueError("All SPHERE activation sets must have the same sample count.")
    indices = deterministic_sample_indices(
        len(candidate_activations), config.sphere_sample_size, config.sphere_seed
    )
    energies = [
        _hyperspherical_energy_for_indices(
            activations,
            indices,
            config.sphere_chunk_size,
            config.sphere_exponent,
        )
        for activations in (
            candidate_activations,
            last_safe_activations,
            original_activations,
        )
    ]
    candidate_energy, last_safe_energy, original_energy = energies
    step_drift = abs(candidate_energy - last_safe_energy) / max(
        abs(last_safe_energy), 1e-12
    )
    cumulative_drift = abs(candidate_energy - original_energy) / max(
        abs(original_energy), 1e-12
    )
    finite = all(isfinite(value) for value in (*energies, step_drift, cumulative_drift))
    safe = finite and not any((
        step_drift >= config.sphere_step_hard_limit,
        cumulative_drift >= config.sphere_cumulative_hard_limit,
    ))
    warning = not finite or any((
        step_drift >= config.sphere_step_warning,
        cumulative_drift >= config.sphere_cumulative_warning,
    ))
    return SphereDriftResult(
        safe=safe,
        warning=warning,
        sample_count=len(indices),
        candidate_energy=candidate_energy,
        last_safe_energy=last_safe_energy,
        original_energy=original_energy,
        step_relative_drift=step_drift,
        cumulative_relative_drift=cumulative_drift,
    )


def assess_consolidation_safety(
    numerical: NumericalGuardResult,
    behavioral: BehavioralDriftResult,
    retention: RetentionResult,
    sphere: SphereDriftResult,
    config: MnemoSafetyConfig,
) -> ConsolidationSafetyResult:
    results = {
        "numerical": numerical,
        "behavioral": behavioral,
        "retention": retention,
        "sphere": sphere,
    }
    hard_failures = [name for name, result in results.items() if not result.safe]
    warning_count = sum(bool(result.warning) for result in results.values())
    warning_failure = warning_count >= config.warning_votes_required
    reasons = list(hard_failures)
    if warning_failure and not hard_failures:
        reasons.append(f"warning_votes:{warning_count}")
    safe = not hard_failures and not warning_failure
    return ConsolidationSafetyResult(
        safe=safe,
        reasons=tuple(reasons),
        warning_count=warning_count,
        numerical=numerical,
        behavioral=behavioral,
        retention=retention,
        sphere=sphere,
        metrics={
            "maximum_relative_update_norm": numerical.maximum_relative_update_norm,
            "step_kl": behavioral.step_kl,
            "cumulative_kl": behavioral.cumulative_kl,
            "step_nll_change": behavioral.step_nll_change,
            "cumulative_nll_change": behavioral.cumulative_nll_change,
            "current_batch_success": retention.current_batch_success,
            "retained_success": retention.retained_success,
            "sphere_step_relative_drift": sphere.step_relative_drift,
            "sphere_cumulative_relative_drift": sphere.cumulative_relative_drift,
        },
    )
