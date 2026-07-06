import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .io import build_method_dir, build_milestone_result_path, build_result_path


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


def _score_from_metric(metric_value: Any) -> float | None:
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


def _last_non_none(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _final_milestone_value(values: list[float | None]) -> float | None:
    if not values:
        return None
    return values[-1]

def _valid_numbers(values) -> list[float]:
    return [float(value) for value in values if value is not None]


def _zero_based_upper(values, pad_ratio: float = 0.08, min_upper: float = 0.05) -> float:
    valid_values = _valid_numbers(values)
    if not valid_values:
        return min_upper
    max_value = max(valid_values)
    if max_value <= 0:
        return min_upper
    return max(max_value * (1 + pad_ratio), min_upper)


def _format_seconds(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "-"


def _format_mean_std(
    value: float | None,
    std: float | None = None,
    decimals: int = 4,
    suffix: str = "",
) -> str:
    if value is None:
        return "-"
    if std is None:
        return f"{value:.{decimals}f}{suffix}"
    return f"{value:.{decimals}f} +/- {std:.{decimals}f}{suffix}"


def _format_seconds_mean_std(value: float | None, std: float | None = None) -> str:
    return _format_mean_std(value, std, decimals=2, suffix="s")


def _ordered_method_names(data: dict[str, Any], methods: list[str]) -> list[str]:
    return [method_name for method_name in methods if method_name in data]


def _print_skipped(skipped: dict[str, str]) -> None:
    if not skipped:
        return
    details = ", ".join(f"{method} ({reason})" for method, reason in skipped.items())
    print(f"Skipped methods due to incomplete data: {details}")


# Result layout: data/evaluation/{model}/{milestones}/r{n_runs}-j{judge_repeats}/{method}/


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


def _load_json_if_exists(path: Path):
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


def _parse_memory_payload(payload) -> dict[str, float | None]:
    if payload is None:
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}

    edit_payload = payload.get("edit", payload) if isinstance(payload, dict) else {}
    if not isinstance(edit_payload, dict):
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}

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

    return {
        "peak_allocated_gb": float(allocated) if allocated is not None else None,
        "peak_reserved_gb": float(reserved) if reserved is not None else None,
    }


def _load_run_summary(
    model: str,
    method: str,
    milestones: list[int],
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    method_dir = build_method_dir(
        edited_model=model,
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
        _parse_memory_payload(payload)
        for payload in (es_loc_memory, port_memory)
        if payload is not None
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
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]], milestones: list[int]) -> dict[str, Any]:
    keys = ("success", "locality", "portability", "avg_time")
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
    model: str,
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
                model=model,
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
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    data, _ = _load_results_with_skips(
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    return data


def _has_complete_method_data(
    model: str,
    method: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> bool:
    return all(
        _load_run_summary(
            model=model,
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
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> list[str]:
    if "base" in methods:
        return methods
    if _has_complete_method_data(
        model=model,
        method="base",
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    ):
        return ["base", *methods]
    return methods


def _line_style_for_method(method_name: str) -> dict[str, Any]:
    if method_name == "base":
        return {"linestyle": "--", "linewidth": 1.9, "alpha": 0.9}
    return {"linestyle": "-", "linewidth": 1.6}


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


def _plot_std_band(
    ax,
    xs: list[int],
    values: list[float | None],
    stds: list[float | None],
    color,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    points = [
        (x, float(value), float(std))
        for x, value, std in zip(xs, values, stds)
        if value is not None and std is not None
    ]
    if not points:
        return

    band_xs = [point[0] for point in points]
    lower = [point[1] - point[2] for point in points]
    upper = [point[1] + point[2] for point in points]
    if lower_bound is not None:
        lower = [max(lower_bound, value) for value in lower]
    if upper_bound is not None:
        upper = [min(upper_bound, value) for value in upper]
    ax.fill_between(band_xs, lower, upper, color=color, alpha=0.14, linewidth=0)


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


def table_perf_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    perf_methods = _performance_methods_with_base(
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    data, skipped = _load_results_with_skips(
        model=model,
        milestones=milestones,
        methods=perf_methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    table_metrics = METRIC_ORDER + [("H", "quality")]
    columns = pd.MultiIndex.from_tuples(
        [(f"step {step}", metric_short) for step in milestones for metric_short, _ in table_metrics]
    )

    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, perf_methods):
        summary = data[method_name]
        row: list[str] = []
        for step_index, _ in enumerate(milestones):
            for _, metric_name in METRIC_ORDER:
                value = summary[metric_name][step_index]
                std = summary.get(f"{metric_name}_std", [None] * len(milestones))[step_index]
                row.append(_format_mean_std(value, std))
            quality_value = _quality_series(summary)[step_index]
            quality_std = _std_series_for_metric(summary, "quality")[step_index]
            row.append(_format_mean_std(quality_value, quality_std))
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"performance | H=harmonic mean(ES, Loc, Port) | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def table_time_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        summary = data[method_name]
        table_data[method_name] = [
            _format_seconds_mean_std(value, std)
            for value, std in zip(summary["avg_time"], summary.get("avg_time_std", [None] * len(milestones)))
        ]

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"avg edit time | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def table_composite_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    final_data = _load_composite_finals(data)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones] + ["Final"]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        composite_values = _composite_series(data[method_name])
        composite_stds = _std_series_for_metric(data[method_name], "composite")
        row = [
            _format_mean_std(value, std)
            for value, std in zip(composite_values, composite_stds)
        ]
        final_value = final_data[method_name]
        final_std = _final_milestone_value(composite_stds)
        row.append(_format_mean_std(final_value, final_std))
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"composite | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def _method_colors(methods_sorted: list[str]) -> dict[str, Any]:
    cmap = plt.colormaps.get_cmap("tab10")
    return {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }


def plot_perf_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (14.0, 17.5),
) -> None:
    milestones = _normalize_milestones(milestones)
    perf_methods = _performance_methods_with_base(
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    data, skipped = _load_results_with_skips(model, milestones, perf_methods, n_runs, judge_repeats, evaluation_dir)
    final_data = _load_performance_finals(data)
    _print_skipped(skipped)
    if not data or not final_data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, perf_methods)
    method_colors = _method_colors(methods_sorted)
    quality_series_by_method = {method_name: _quality_series(data[method_name]) for method_name in methods_sorted}
    plot_metrics = METRIC_ORDER + [("H", "quality")]
    perf_bar_upper = _zero_based_upper(
        value
        for method_name in methods_sorted
        for value in (
            list(final_data[method_name].values())
            + [_final_milestone_value(quality_series_by_method[method_name])]
        )
    )

    fig, axes = plt.subplots(len(plot_metrics), 2, figsize=figsize, squeeze=False)
    for metric_index, (short_name, metric_name) in enumerate(plot_metrics):
        line_ax = axes[metric_index][0]
        bar_ax = axes[metric_index][1]
        has_any_line = False

        for method_name in methods_sorted:
            values = _series_for_metric(data[method_name], metric_name)
            stds = _std_series_for_metric(data[method_name], metric_name)
            if not any(value is not None for value in values):
                continue
            _plot_std_band(line_ax, milestones, values, stds, method_colors[method_name], lower_bound=0.0, upper_bound=1.0)
            line_ax.plot(
                milestones,
                values,
                marker="o",
                markersize=4,
                color=method_colors[method_name],
                **_line_style_for_method(method_name),
            )
            has_any_line = True

        if not has_any_line:
            line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

        line_ax.set_ylim(-0.05, 1.05)
        line_ax.set_xticks(milestones)
        line_ax.set_xlabel("step")
        line_ax.set_ylabel("score")
        line_title = "H-Mean Sequential" if metric_name == "quality" else f"{short_name} Sequential"
        line_ax.set_title(line_title, fontweight="bold")
        line_ax.grid(alpha=0.25)

        values, stds = _final_values_and_stds(data, methods_sorted, metric_name)
        bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
        bar_values = [value for value in values if value is not None]
        bar_stds = [
            std
            for value, std in zip(values, stds)
            if value is not None and std is not None
        ]
        yerr = bar_stds if len(bar_stds) == len(bar_values) else None
        if bar_values:
            bar_ax.bar(
                bar_methods,
                bar_values,
                yerr=yerr,
                capsize=4 if yerr is not None else 0,
                color=[method_colors[method] for method in bar_methods],
                width=0.7,
            )
        else:
            bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
            bar_ax.set_xticks([])
        bar_ax.tick_params(axis="x", rotation=45)
        bar_ax.set_ylabel("final score")
        bar_ax.set_ylim(0, perf_bar_upper)
        bar_title = "H-Mean Final" if metric_name == "quality" else f"{short_name} Final"
        bar_ax.set_title(f"{bar_title} @ {milestones[-1]}", fontweight="bold")
        bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=method_colors[method],
                lw=_line_style_for_method(method).get("linewidth", 2),
                linestyle=_line_style_for_method(method).get("linestyle", "-"),
                label=method,
            )
            for method in methods_sorted
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="method",
    )
    fig.suptitle(f"performance | model={model} milestones={milestones} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_time_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> None:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    method_colors = _method_colors(methods_sorted)
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    line_ax, bar_ax = axes[0]

    has_any_line = False
    for method_name in methods_sorted:
        values = data[method_name]["avg_time"]
        stds = data[method_name].get("avg_time_std", [None] * len(milestones))
        if not any(value is not None for value in values):
            continue
        _plot_std_band(line_ax, milestones, values, stds, method_colors[method_name], lower_bound=0.0)
        line_ax.plot(milestones, values, marker="o", markersize=4, linewidth=1.6, color=method_colors[method_name])
        has_any_line = True

    if not has_any_line:
        line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

    line_ax.set_xticks(milestones)
    line_ax.set_xlabel("step")
    line_ax.set_ylabel("avg edit time so far (s)")
    line_ax.set_ylim(0, _zero_based_upper(value for method_name in methods_sorted for value in data[method_name]["avg_time"]))
    line_ax.set_title("Sequential Avg Edit Time", fontweight="bold")
    line_ax.grid(alpha=0.25)

    final_times = [_final_milestone_value(data[method_name]["avg_time"]) for method_name in methods_sorted]
    final_time_stds = [
        _final_milestone_value(data[method_name].get("avg_time_std", [None] * len(milestones)))
        for method_name in methods_sorted
    ]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, final_times) if value is not None]
    bar_values = [value for value in final_times if value is not None]
    bar_stds = [
        std
        for value, std in zip(final_times, final_time_stds)
        if value is not None and std is not None
    ]
    yerr = bar_stds if len(bar_stds) == len(bar_values) else None
    if bar_values:
        bar_ax.bar(
            bar_methods,
            bar_values,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            color=[method_colors[method] for method in bar_methods],
            width=0.7,
        )
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])
    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("final avg edit time (s)")
    bar_ax.set_title("Final Avg Edit Time", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=method_colors[method],
                lw=_line_style_for_method(method).get("linewidth", 2),
                linestyle=_line_style_for_method(method).get("linestyle", "-"),
                label=method,
            )
            for method in methods_sorted
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="method",
    )
    fig.suptitle(f"time | model={model} milestones={milestones} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_composite_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> None:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    final_data = _load_composite_finals(data)
    _print_skipped(skipped)
    if not data or not final_data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    method_colors = _method_colors(methods_sorted)
    composite_series_by_method = {method_name: _composite_series(data[method_name]) for method_name in methods_sorted}

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    line_ax, bar_ax = axes[0]

    has_any_line = False
    for method_name in methods_sorted:
        values = composite_series_by_method[method_name]
        stds = _std_series_for_metric(data[method_name], "composite")
        if not any(value is not None for value in values):
            continue
        _plot_std_band(line_ax, milestones, values, stds, method_colors[method_name], lower_bound=0.0)
        line_ax.plot(milestones, values, marker="o", markersize=4, linewidth=1.6, color=method_colors[method_name])
        has_any_line = True

    if not has_any_line:
        line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

    line_ax.set_ylim(0, _zero_based_upper(value for values in composite_series_by_method.values() for value in values))
    line_ax.set_xticks(milestones)
    line_ax.set_xlabel("step")
    line_ax.set_ylabel("composite score")
    line_ax.set_title("Composite Sequential", fontweight="bold")
    line_ax.grid(alpha=0.25)

    values = [final_data[method_name] for method_name in methods_sorted]
    stds = [
        _final_milestone_value(_std_series_for_metric(data[method_name], "composite"))
        for method_name in methods_sorted
    ]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
    bar_values = [value for value in values if value is not None]
    bar_stds = [
        std
        for value, std in zip(values, stds)
        if value is not None and std is not None
    ]
    yerr = bar_stds if len(bar_stds) == len(bar_values) else None
    if bar_values:
        bar_ax.bar(
            bar_methods,
            bar_values,
            yerr=yerr,
            capsize=4 if yerr is not None else 0,
            color=[method_colors[method] for method in bar_methods],
            width=0.7,
        )
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])
    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("final composite score")
    bar_ax.set_ylim(0, _zero_based_upper(bar_values))
    bar_ax.set_title(f"Composite Final @ {milestones[-1]}", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=method_colors[method],
                lw=_line_style_for_method(method).get("linewidth", 2),
                linestyle=_line_style_for_method(method).get("linestyle", "-"),
                label=method,
            )
            for method in methods_sorted
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="method",
    )
    fig.suptitle(f"composite | model={model} milestones={milestones} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def table_memory_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        summary = data[method_name]
        table_data[method_name] = [
            _format_mean_std(
                summary.get("peak_allocated_gb"),
                summary.get("peak_allocated_gb_std"),
                decimals=2,
                suffix=" GB",
            ),
            _format_mean_std(
                summary.get("peak_reserved_gb"),
                summary.get("peak_reserved_gb_std"),
                decimals=2,
                suffix=" GB",
            ),
        ]

    df = pd.DataFrame.from_dict(
        table_data,
        orient="index",
        columns=["peak allocated", "peak reserved"],
    )
    df.index.name = "method"
    print(f"peak edit memory | model={model} | r{n_runs}-j{judge_repeats}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def plot_memory_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
) -> None:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    method_colors = _method_colors(methods_sorted)
    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)

    for ax, key, title in [
        (axes[0][0], "peak_allocated_gb", "Peak Allocated"),
        (axes[0][1], "peak_reserved_gb", "Peak Reserved"),
    ]:
        values = [data[method].get(key) for method in methods_sorted]
        stds = [data[method].get(f"{key}_std") for method in methods_sorted]
        bar_methods = [method for method, value in zip(methods_sorted, values) if value is not None]
        bar_values = [value for value in values if value is not None]
        bar_stds = [
            std
            for value, std in zip(values, stds)
            if value is not None and std is not None
        ]
        yerr = bar_stds if len(bar_stds) == len(bar_values) else None

        if bar_values:
            ax.bar(
                bar_methods,
                bar_values,
                yerr=yerr,
                capsize=4 if yerr is not None else 0,
                color=[method_colors[method] for method in bar_methods],
                width=0.7,
            )
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_xticks([])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("GB")
        ax.set_ylim(0, _zero_based_upper(bar_values, min_upper=1.0))
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"peak edit memory | model={model} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_pareto_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    method_colors = _method_colors(methods_sorted)
    memory_values = [data[method].get("peak_allocated_gb") for method in methods_sorted]
    valid_memory = _valid_numbers(memory_values)
    min_mem = min(valid_memory) if valid_memory else None
    max_mem = max(valid_memory) if valid_memory else None

    def point_size(memory_value: float | None) -> float:
        if memory_value is None or min_mem is None or max_mem is None or max_mem <= min_mem:
            return 95.0
        return 70.0 + 190.0 * ((memory_value - min_mem) / (max_mem - min_mem))

    fig, ax = plt.subplots(figsize=figsize)
    has_points = False
    for method_name in methods_sorted:
        summary = data[method_name]
        avg_time = _final_milestone_value(summary["avg_time"])
        avg_time_std = _final_milestone_value(summary.get("avg_time_std", [None] * len(milestones)))
        quality = _final_milestone_value(_quality_series(summary))
        quality_std = _final_milestone_value(_std_series_for_metric(summary, "quality"))
        memory = summary.get("peak_allocated_gb")
        if avg_time is None or quality is None:
            continue

        ax.errorbar(
            avg_time,
            quality,
            xerr=avg_time_std,
            yerr=quality_std,
            fmt="none",
            ecolor=method_colors[method_name],
            alpha=0.45,
            capsize=3,
        )
        ax.scatter(
            avg_time,
            quality,
            s=point_size(memory),
            color=method_colors[method_name],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.5,
            label=method_name,
        )
        ax.annotate(method_name, (avg_time, quality), xytext=(5, 4), textcoords="offset points", fontsize=9)
        has_points = True

    if not has_points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")

    ax.set_xlabel("final avg edit time (s)")
    ax.set_ylabel("final harmonic score")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_title(f"Quality/Time Pareto @ {milestones[-1]}", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", title="method")
    fig.suptitle(f"pareto | point size = peak allocated memory | model={model} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def audit_result_grid(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    result_names: list[str] | None = None,
    result_groups: list[str] | None = None,
    evaluation_dir: Path | str | None = None,
    only_problems: bool = True,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    result_names = result_names or ["es_loc", "port"]
    result_groups = result_groups or ["unscored", "scored", "perf", "time", "mem"]
    runs = _runs_from_count(n_runs)
    rows: list[dict[str, Any]] = []

    for method_name in methods:
        for result_group in result_groups:
            for result_name in result_names:
                for run_index in runs:
                    if result_group in {"unscored", "scored"}:
                        for step in milestones:
                            path = build_milestone_result_path(
                                edited_model=model,
                                editing_method=method_name,
                                milestones=milestones,
                                result_name=result_name,
                                run_index=run_index,
                                step=step,
                                n_runs=n_runs,
                                judge_repeats=judge_repeats,
                                result_group=result_group,
                                evaluation_dir=evaluation_dir,
                            )
                            exists = path.exists()
                            size = path.stat().st_size if exists else None
                            status = "present" if exists and size else "empty" if exists else "missing"
                            rows.append({
                                "method": method_name,
                                "group": result_group,
                                "result": result_name,
                                "run": run_index,
                                "step": step,
                                "status": status,
                                "path": str(path),
                            })
                    else:
                        path = build_result_path(
                            edited_model=model,
                            editing_method=method_name,
                            milestones=milestones,
                            result_name=result_name,
                            run_index=run_index,
                            n_runs=n_runs,
                            judge_repeats=judge_repeats,
                            result_group=result_group,
                            evaluation_dir=evaluation_dir,
                        )
                        exists = path.exists()
                        size = path.stat().st_size if exists else None
                        status = "present" if exists and size else "empty" if exists else "missing"
                        rows.append({
                            "method": method_name,
                            "group": result_group,
                            "result": result_name,
                            "run": run_index,
                            "step": None,
                            "status": status,
                            "path": str(path),
                        })

    df = pd.DataFrame(rows)
    summary = df.groupby(["method", "group", "status"]).size().unstack(fill_value=0)
    print(f"result-grid audit | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    if not summary.empty:
        print(summary)

    if only_problems:
        problem_df = df[df["status"] != "present"].reset_index(drop=True)
        if problem_df.empty:
            print("No missing or empty files found for the selected grid.")
        return problem_df
    return df


