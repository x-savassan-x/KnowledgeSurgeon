import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .io import build_method_dir


METRIC_ORDER: list[tuple[str, str]] = [
    ("ES", "success"),
    ("Loc", "locality"),
    ("Port", "portability"),
]

EPS = 1e-9


def _normalize_milestones(milestones: list[int]) -> list[int]:
    normalized = sorted({int(step) for step in milestones})
    if not normalized:
        raise ValueError("milestones must not be empty")
    if normalized[0] <= 0:
        raise ValueError("milestones must be positive integers")
    return normalized


def _runs_from_count(n_runs: int) -> list[int]:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    return list(range(1, n_runs + 1))


def _score_from_metric(metric_value) -> float | None:
    if metric_value is None:
        return None
    if isinstance(metric_value, dict):
        score = metric_value.get("score")
        if score is None:
            return None
        return float(score)
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _mean_ignore_none(values: list[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return mean(valid_values)


def _std_ignore_none(values: list[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    if len(valid_values) == 1:
        return 0.0
    return stdev(valid_values)


def _final_milestone_value(values: list[float | None]) -> float | None:
    if not values:
        return None
    return values[-1]


def _valid_numbers(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _zero_based_upper(
    values: list[Any],
    pad_ratio: float = 0.08,
    min_upper: float = 0.05,
) -> float:
    valid_values = _valid_numbers(values)
    if not valid_values:
        return min_upper
    max_value = max(valid_values)
    if max_value <= 0:
        return min_upper
    return max(max_value * (1 + pad_ratio), min_upper)


def _format_mean_std(
    value: float | None,
    std: float | None = None,
    decimals: int = 4,
    suffix: str = "",
    show_std: bool = True,
) -> str:
    if value is None:
        return "-"
    if not show_std or std is None:
        return f"{value:.{decimals}f}{suffix}"
    return f"{value:.{decimals}f} +/- {std:.{decimals}f}{suffix}"


def _format_seconds_mean_std(
    value: float | None,
    std: float | None = None,
    show_std: bool = True,
) -> str:
    return _format_mean_std(value, std, decimals=2, suffix="s", show_std=show_std)


def _ordered_method_names(data: dict[str, Any], methods: list[str]) -> list[str]:
    return [method_name for method_name in methods if method_name in data]


def _print_skipped(skipped: dict[str, str]) -> None:
    if not skipped:
        return
    details = ", ".join(f"{method} ({reason})" for method, reason in skipped.items())
    print(f"Skipped methods due to incomplete data: {details}")


def _strict_harmonic_mean(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return None
    return len(valid_values) / sum(1 / (value + EPS) for value in valid_values)


def _extract_score_series(
    records: list[dict[str, Any]],
    metric_name: str,
    milestones: list[int],
) -> list[float | None]:
    score_by_step: dict[int, float | None] = {step: None for step in milestones}
    for record in records:
        step = record.get("step")
        if not isinstance(step, int) or step not in score_by_step:
            continue
        result = record.get("result", record)
        if not isinstance(result, dict):
            continue
        score = _score_from_metric(result.get(metric_name))
        if score is not None:
            score_by_step[step] = score
    return [score_by_step[step] for step in milestones]


def _avg_time_series_from_edit_times(edit_times: list[float], milestones: list[int]) -> list[float | None]:
    series: list[float | None] = []
    for step in milestones:
        if step <= 0 or step > len(edit_times):
            series.append(None)
            continue
        series.append(sum(edit_times[:step]) / step)
    return series


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _parse_time_payload(payload, milestones: list[int]) -> dict[str, list[float | None]]:
    if payload is None:
        return {"avg_time": [None] * len(milestones)}
    if isinstance(payload, list):
        edit_times = [float(value) for value in payload]
    else:
        edit_times = [float(value) for value in payload.get("edit", [])]

    return {"avg_time": _avg_time_series_from_edit_times(edit_times, milestones)}


def _parse_memory_payload(
    payload,
    milestones: list[int],
) -> dict[str, Any]:
    empty_series = [None] * len(milestones)
    if payload is None:
        return {
            "allocated_gb": empty_series.copy(),
            "reserved_gb": empty_series.copy(),
            "peak_allocated_gb": None,
            "peak_reserved_gb": None,
        }

    edit_payload = payload.get("edit", payload) if isinstance(payload, dict) else {}
    if not isinstance(edit_payload, dict):
        return {
            "allocated_gb": empty_series.copy(),
            "reserved_gb": empty_series.copy(),
            "peak_allocated_gb": None,
            "peak_reserved_gb": None,
        }

    allocated = edit_payload.get("total_peak_allocated_gb")
    reserved = edit_payload.get("total_peak_reserved_gb")
    if allocated is None or reserved is None:
        devices = edit_payload.get("devices", [])
        if isinstance(devices, list):
            allocated_values = [
                float(device["peak_allocated_gb"])
                for device in devices
                if isinstance(device, dict) and device.get("peak_allocated_gb") is not None
            ]
            reserved_values = [
                float(device["peak_reserved_gb"])
                for device in devices
                if isinstance(device, dict) and device.get("peak_reserved_gb") is not None
            ]
            allocated = sum(allocated_values) if allocated_values else allocated
            reserved = sum(reserved_values) if reserved_values else reserved

    allocated_by_step = {step: None for step in milestones}
    reserved_by_step = {step: None for step in milestones}
    for milestone in edit_payload.get("milestones", []):
        if not isinstance(milestone, dict):
            continue
        step = milestone.get("step")
        if step not in allocated_by_step:
            continue

        milestone_allocated = milestone.get("total_allocated_gb")
        milestone_reserved = milestone.get("total_reserved_gb")
        devices = milestone.get("devices", [])
        if isinstance(devices, list):
            allocated_values = [
                float(device["allocated_gb"])
                for device in devices
                if isinstance(device, dict) and device.get("allocated_gb") is not None
            ]
            reserved_values = [
                float(device["reserved_gb"])
                for device in devices
                if isinstance(device, dict) and device.get("reserved_gb") is not None
            ]
            if milestone_allocated is None and allocated_values:
                milestone_allocated = sum(allocated_values)
            if milestone_reserved is None and reserved_values:
                milestone_reserved = sum(reserved_values)

        allocated_by_step[step] = float(milestone_allocated) if milestone_allocated is not None else None
        reserved_by_step[step] = float(milestone_reserved) if milestone_reserved is not None else None

    return {
        "allocated_gb": [allocated_by_step[step] for step in milestones],
        "reserved_gb": [reserved_by_step[step] for step in milestones],
        "peak_allocated_gb": float(allocated) if allocated is not None else None,
        "peak_reserved_gb": float(reserved) if reserved is not None else None,
    }


def _load_run_summary(
    model_key: str,
    method: str,
    milestones: list[int],
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    method_dir = build_method_dir(
        model_key=model_key,
        editing_method=method,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )

    es_loc_records = _load_json_if_exists(method_dir / "perf" / f"es_loc.{run_index}.json")
    port_records = _load_json_if_exists(method_dir / "perf" / f"port.{run_index}.json")
    es_loc_time = _load_json_if_exists(method_dir / "time" / f"es_loc.{run_index}.json")
    port_time = _load_json_if_exists(method_dir / "time" / f"port.{run_index}.json")
    es_loc_memory = _load_json_if_exists(method_dir / "mem" / f"es_loc.{run_index}.json")
    port_memory = _load_json_if_exists(method_dir / "mem" / f"port.{run_index}.json")

    if es_loc_records is None and port_records is None:
        return None

    time_summaries = [
        _parse_time_payload(payload, milestones)
        for payload in (es_loc_time, port_time)
        if payload is not None
    ]
    if time_summaries:
        avg_time = [
            _mean_ignore_none([summary["avg_time"][idx] for summary in time_summaries])
            for idx in range(len(milestones))
        ]
    else:
        avg_time = [None] * len(milestones)

    memory_summaries = [
        _parse_memory_payload(payload, milestones)
        for payload in (es_loc_memory, port_memory)
        if payload is not None
    ]
    allocated_gb = [
        max(
            (summary["allocated_gb"][idx] for summary in memory_summaries if summary["allocated_gb"][idx] is not None),
            default=None,
        )
        for idx in range(len(milestones))
    ]
    reserved_gb = [
        max(
            (summary["reserved_gb"][idx] for summary in memory_summaries if summary["reserved_gb"][idx] is not None),
            default=None,
        )
        for idx in range(len(milestones))
    ]
    peak_allocated_gb = (
        max(
            (summary["peak_allocated_gb"] for summary in memory_summaries if summary["peak_allocated_gb"] is not None),
            default=None,
        )
        if memory_summaries
        else None
    )
    peak_reserved_gb = (
        max(
            (summary["peak_reserved_gb"] for summary in memory_summaries if summary["peak_reserved_gb"] is not None),
            default=None,
        )
        if memory_summaries
        else None
    )

    return {
        "success": _extract_score_series(es_loc_records or [], "success", milestones),
        "locality": _extract_score_series(es_loc_records or [], "locality", milestones),
        "portability": _extract_score_series(port_records or [], "portability", milestones),
        "avg_time": avg_time,
        "allocated_gb": allocated_gb,
        "reserved_gb": reserved_gb,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]], milestones: list[int]) -> dict[str, Any]:
    keys = ("success", "locality", "portability", "avg_time", "allocated_gb", "reserved_gb")
    aggregate: dict[str, Any] = {"runs": run_summaries}
    for key in keys:
        aggregate[key] = [
            _mean_ignore_none([summary[key][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ]
        aggregate[f"{key}_std"] = [
            _std_ignore_none([summary[key][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ]

    quality_runs = [_quality_series(summary) for summary in run_summaries]
    aggregate["quality"] = [
        _mean_ignore_none([series[idx] for series in quality_runs])
        for idx in range(len(milestones))
    ]
    aggregate["quality_std"] = [
        _std_ignore_none([series[idx] for series in quality_runs])
        for idx in range(len(milestones))
    ]

    composite_runs = [_composite_series(summary) for summary in run_summaries]
    aggregate["composite"] = [
        _mean_ignore_none([series[idx] for series in composite_runs])
        for idx in range(len(milestones))
    ]
    aggregate["composite_std"] = [
        _std_ignore_none([series[idx] for series in composite_runs])
        for idx in range(len(milestones))
    ]

    for key in ("peak_allocated_gb", "peak_reserved_gb"):
        values = [summary.get(key) for summary in run_summaries]
        aggregate[key] = _mean_ignore_none(values)
        aggregate[f"{key}_std"] = _std_ignore_none(values)

    return aggregate


def _load_results_with_skips(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)

    data: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for method_name in methods:
        run_summaries: list[dict[str, Any]] = []
        missing_runs: list[int] = []
        for run_index in runs:
            summary = _load_run_summary(
                model_key=model_key,
                method=method_name,
                milestones=milestones,
                run_index=run_index,
                n_runs=n_runs,
                judge_repeats=judge_repeats,
                evaluation_dir=evaluation_dir,
            )
            if summary is None:
                missing_runs.append(run_index)
            else:
                run_summaries.append(summary)

        if missing_runs:
            skipped[method_name] = f"missing runs {missing_runs}"
            continue
        data[method_name] = _aggregate_runs(run_summaries, milestones)

    return data, skipped


def load_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    data, _ = _load_results_with_skips(
        model_key=model_key,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    return data


def _has_complete_method_data(
    model_key: str,
    method: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> bool:
    return all(
        _load_run_summary(
            model_key=model_key,
            method=method,
            milestones=milestones,
            run_index=run_index,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        ) is not None
        for run_index in _runs_from_count(n_runs)
    )


def _performance_methods_with_base(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> list[str]:
    if "base" in methods:
        return methods
    if _has_complete_method_data(
        model_key=model_key,
        method="base",
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    ):
        return ["base", *methods]
    return methods


def _load_performance_finals(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        method_name: {
            metric_name: _final_milestone_value(summary[metric_name])
            for _, metric_name in METRIC_ORDER
        }
        for method_name, summary in data.items()
    }


def _quality_series(summary: dict[str, Any]) -> list[float | None]:
    if "quality" in summary:
        return summary["quality"]
    return [
        _strict_harmonic_mean([
            summary["success"][idx],
            summary["locality"][idx],
            summary["portability"][idx],
        ])
        for idx in range(len(summary["success"]))
    ]


def _composite_series(summary: dict[str, Any]) -> list[float | None]:
    if "composite" in summary:
        return summary["composite"]
    quality_series = _quality_series(summary)
    return [
        None if quality is None or avg_time is None else quality / (sqrt(avg_time) + EPS)
        for quality, avg_time in zip(quality_series, summary["avg_time"])
    ]


def _load_composite_finals(data: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {
        method_name: _final_milestone_value(_composite_series(summary))
        for method_name, summary in data.items()
    }


def _series_for_metric(summary: dict[str, Any], metric_name: str) -> list[float | None]:
    if metric_name == "quality":
        return _quality_series(summary)
    if metric_name == "composite":
        return _composite_series(summary)
    return summary[metric_name]


def _std_series_for_metric(summary: dict[str, Any], metric_name: str) -> list[float | None]:
    if metric_name == "quality":
        return summary.get("quality_std", [None] * len(_quality_series(summary)))
    if metric_name == "composite":
        return summary.get("composite_std", [None] * len(_composite_series(summary)))
    return summary.get(f"{metric_name}_std", [None] * len(summary[metric_name]))


def _final_values_and_stds(
    data: dict[str, dict[str, Any]],
    methods_sorted: list[str],
    metric_name: str,
) -> tuple[list[float | None], list[float | None]]:
    values = [
        _final_milestone_value(_series_for_metric(data[method_name], metric_name))
        for method_name in methods_sorted
    ]
    stds = [
        _final_milestone_value(_std_series_for_metric(data[method_name], metric_name))
        for method_name in methods_sorted
    ]
    return values, stds
