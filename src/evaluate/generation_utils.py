from __future__ import annotations

import copy
from itertools import chain
from typing import Any

from .schema import EvalRecord, EvalResult


def flatten_record_groups(records: list[list[EvalRecord]]) -> list[EvalRecord]:
    return list(chain.from_iterable(records))


def get_generation_runtime_config(
    generation_config: dict[str, Any],
    desc: str,
    editing_method: str | None = None,
) -> dict[str, Any]:
    runtime_config = copy.deepcopy(generation_config)
    method_overrides = runtime_config.pop("method_overrides", {}) or {}
    runtime_config.pop("async_generation", None)
    if editing_method in method_overrides:
        override = copy.deepcopy(method_overrides[editing_method] or {})
        override_model_config = override.pop("model_config", None)
        if override_model_config:
            runtime_config.setdefault("model_config", {}).update(override_model_config)
        runtime_config.update(override)
    runtime_config["desc"] = desc
    return runtime_config


def clean_prediction(
    prediction: str,
    stop_strings: list[str],
    strip_leading_newlines: bool = True,
    truncate_at_stop_strings: bool = True,
) -> str:
    text = prediction
    if strip_leading_newlines:
        text = text.lstrip(" \t\r\n")

    if truncate_at_stop_strings:
        stop_positions = [
            text.find(stop_string)
            for stop_string in stop_strings
            if stop_string and text.find(stop_string) >= 0
        ]
        if stop_positions:
            text = text[:min(stop_positions)]

    return text.strip()


def snapshot_eval_inputs(
    eval_results: list[EvalResult],
    step: int,
) -> tuple[list[EvalResult], list[list[EvalRecord]]]:
    snapshot_results = copy.deepcopy(eval_results[:step])
    snapshot_records: list[list[EvalRecord]] = []
    for result in snapshot_results:
        details: list[EvalRecord] = []
        for metric_result in (result.success, result.locality, result.portability):
            if metric_result is not None:
                details.extend(metric_result.details)
        snapshot_records.append(details)
    return snapshot_results, snapshot_records


def edit_requests_to_payload(edit_requests: list[Any], step: int) -> list[dict[str, Any]]:
    return [
        {
            "prompt": request.prompt,
            "old_ground_truth": request.old_ground_truth,
            "new_ground_truth": request.new_ground_truth,
        }
        for request in edit_requests[:step]
    ]
