"""Deterministic cohorts and orderings for sensitivity evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_identity(record: Mapping[str, Any], result_name: str) -> dict[str, str]:
    """Return the stable edit identity used for cohort/order fingerprints."""
    edit = record["edit"]
    answer = edit["answer"]["new"]["str"]
    subject = edit["subject"]
    if isinstance(subject, Mapping):
        subject = subject.get("str", subject)
    return {
        "result": result_name,
        "subject": str(subject),
        "prompt": str(edit["prompt"]),
        "answer": str(answer),
    }


def fingerprint_records(
    records: Sequence[Mapping[str, Any]],
    result_name: str,
) -> str:
    identities = [record_identity(record, result_name) for record in records]
    return _json_hash(identities)


def sensitivity_run_indices(n_runs: int, run_start: int = 1, run_count: int | None = None) -> list[int]:
    """Return the selected one-based order indices for a sensitivity run."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if run_start <= 0:
        raise ValueError("run_start must be positive")
    if run_count is None:
        run_count = n_runs - run_start + 1
    if run_count <= 0 or run_start + run_count - 1 > n_runs:
        raise ValueError("requested sensitivity run range is outside n_runs")
    return list(range(run_start, run_start + run_count))


def select_fixed_cohort(
    dataset: Sequence[Mapping[str, Any]],
    result_name: str,
    size: int,
    base_seed: int,
    dataset_offset: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Select one reproducible cohort independently for each result type."""
    if size <= 0:
        raise ValueError("cohort size must be positive")
    if size > len(dataset):
        raise ValueError(f"Requested {size} records, but only {len(dataset)} are available")
    records = list(dataset)
    seed = int(base_seed) + int(dataset_offset)
    Random(seed).shuffle(records)
    cohort = records[:size]
    return cohort, {
        "result": result_name,
        "seed": seed,
        "size": size,
        "fingerprint": fingerprint_records(cohort, result_name),
    }


def order_cohort(
    cohort: Sequence[Mapping[str, Any]],
    result_name: str,
    base_seed: int,
    run_index: int,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Create a deterministic permutation of an already selected cohort."""
    if run_index <= 0:
        raise ValueError("run_index must be positive")
    seed = int(base_seed) + int(run_index) - 1
    ordered = list(cohort)
    Random(seed).shuffle(ordered)
    return ordered, {
        "result": result_name,
        "run_index": int(run_index),
        "seed": seed,
        "size": len(ordered),
        "fingerprint": fingerprint_records(ordered, result_name),
    }
