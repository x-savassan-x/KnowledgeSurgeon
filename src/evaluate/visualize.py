import json
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

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


def _last_non_none(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


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

    return {
        "success": _extract_score_series(es_loc_records or [], "success", milestones),
        "locality": _extract_score_series(es_loc_records or [], "locality", milestones),
        "portability": _extract_score_series(port_records or [], "portability", milestones),
        "avg_time": avg_time,
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]], milestones: list[int]) -> dict[str, Any]:
    keys = ("success", "locality", "portability", "avg_time")
    return {
        key: [
            _mean_ignore_none([summary[key][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ]
        for key in keys
    }


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


def _load_performance_means(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        method_name: {
            metric_name: _mean_ignore_none(summary[metric_name])
            for _, metric_name in METRIC_ORDER
        }
        for method_name, summary in data.items()
    }


def _quality_series(summary: dict[str, Any]) -> list[float | None]:
    return [
        _strict_harmonic_mean([
            summary["success"][idx],
            summary["locality"][idx],
            summary["portability"][idx],
        ])
        for idx in range(len(summary["success"]))
    ]


def _composite_series(summary: dict[str, Any]) -> list[float | None]:
    quality_series = _quality_series(summary)
    return [
        None if quality is None or avg_time is None else quality / (sqrt(avg_time) + EPS)
        for quality, avg_time in zip(quality_series, summary["avg_time"])
    ]


def _load_composite_means(data: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    composite_means: dict[str, float | None] = {}
    for method_name, summary in data.items():
        mean_quality = _strict_harmonic_mean([
            _mean_ignore_none(summary["success"]),
            _mean_ignore_none(summary["locality"]),
            _mean_ignore_none(summary["portability"]),
        ])
        avg_edit_time = _last_non_none(summary["avg_time"])
        if mean_quality is None or avg_edit_time is None:
            composite_means[method_name] = None
        else:
            composite_means[method_name] = mean_quality / (sqrt(avg_edit_time) + EPS)
    return composite_means


def table_perf_results(
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    mean_data = _load_performance_means(data)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = pd.MultiIndex.from_tuples(
        [(f"step {step}", metric_short) for step in milestones for metric_short, _ in METRIC_ORDER]
        + [("Mean", metric_short) for metric_short, _ in METRIC_ORDER]
    )

    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        summary = data[method_name]
        row: list[str] = []
        for step_index, _ in enumerate(milestones):
            for _, metric_name in METRIC_ORDER:
                value = summary[metric_name][step_index]
                row.append(f"{value:.4f}" if value is not None else "-")
        for _, metric_name in METRIC_ORDER:
            value = mean_data[method_name][metric_name]
            row.append(f"{value:.4f}" if value is not None else "-")
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"performance | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
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
        table_data[method_name] = [_format_seconds(value) for value in data[method_name]["avg_time"]]

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
    mean_data = _load_composite_means(data)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones] + ["Mean"]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        composite_values = _composite_series(data[method_name])
        row = [f"{value:.4f}" if value is not None else "-" for value in composite_values]
        mean_value = mean_data[method_name]
        row.append(f"{mean_value:.4f}" if mean_value is not None else "-")
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
    figsize: tuple[float, float] = (14.0, 13.5),
) -> None:
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(model, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    mean_data = _load_performance_means(data)
    _print_skipped(skipped)
    if not data or not mean_data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    method_colors = _method_colors(methods_sorted)
    perf_bar_upper = _zero_based_upper(
        mean_data[method_name][metric_name]
        for method_name in methods_sorted
        for _, metric_name in METRIC_ORDER
    )

    fig, axes = plt.subplots(3, 2, figsize=figsize, squeeze=False)
    for metric_index, (short_name, metric_name) in enumerate(METRIC_ORDER):
        line_ax = axes[metric_index][0]
        bar_ax = axes[metric_index][1]
        has_any_line = False

        for method_name in methods_sorted:
            values = data[method_name][metric_name]
            if not any(value is not None for value in values):
                continue
            line_ax.plot(milestones, values, marker="o", markersize=4, linewidth=1.6, color=method_colors[method_name])
            has_any_line = True

        if not has_any_line:
            line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

        line_ax.set_ylim(-0.05, 1.05)
        line_ax.set_xticks(milestones)
        line_ax.set_xlabel("step")
        line_ax.set_ylabel("score")
        line_ax.set_title(f"{short_name} Sequential", fontweight="bold")
        line_ax.grid(alpha=0.25)

        values = [mean_data[method_name][metric_name] for method_name in methods_sorted]
        bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
        bar_values = [value for value in values if value is not None]
        if bar_values:
            bar_ax.bar(bar_methods, bar_values, color=[method_colors[method] for method in bar_methods], width=0.7)
        else:
            bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
            bar_ax.set_xticks([])
        bar_ax.tick_params(axis="x", rotation=45)
        bar_ax.set_ylabel("mean score")
        bar_ax.set_ylim(0, perf_bar_upper)
        bar_ax.set_title(f"{short_name} Mean", fontweight="bold")
        bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[Line2D([0], [0], color=method_colors[method], lw=2, label=method) for method in methods_sorted],
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
        if not any(value is not None for value in values):
            continue
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

    final_times = [_last_non_none(data[method_name]["avg_time"]) for method_name in methods_sorted]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, final_times) if value is not None]
    bar_values = [value for value in final_times if value is not None]
    if bar_values:
        bar_ax.bar(bar_methods, bar_values, color=[method_colors[method] for method in bar_methods], width=0.7)
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])
    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("final avg edit time (s)")
    bar_ax.set_title("Final Avg Edit Time", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[Line2D([0], [0], color=method_colors[method], lw=2, label=method) for method in methods_sorted],
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
    mean_data = _load_composite_means(data)
    _print_skipped(skipped)
    if not data or not mean_data:
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
        if not any(value is not None for value in values):
            continue
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

    values = [mean_data[method_name] for method_name in methods_sorted]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
    bar_values = [value for value in values if value is not None]
    if bar_values:
        bar_ax.bar(bar_methods, bar_values, color=[method_colors[method] for method in bar_methods], width=0.7)
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])
    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("mean composite score")
    bar_ax.set_ylim(0, _zero_based_upper(bar_values))
    bar_ax.set_title("Composite Mean", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    fig.legend(
        handles=[Line2D([0], [0], color=method_colors[method], lw=2, label=method) for method in methods_sorted],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="method",
    )
    fig.suptitle(f"composite | model={model} milestones={milestones} r{n_runs}-j{judge_repeats}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()
