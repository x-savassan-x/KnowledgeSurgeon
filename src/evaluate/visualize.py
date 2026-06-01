import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .io import EVALUATION_DIR, milestones_to_str


METRIC_ORDER: list[tuple[str, str]] = [
    ("ES", "success"),
    ("Loc", "locality"),
    ("Port", "portability"),
]

EDIT_TIME_KEY = "edit_time"
TOTAL_TIME_KEY = "total_edit_time"


def _normalize_milestones(milestones: list[int]) -> list[int]:
    normalized = sorted({int(step) for step in milestones})
    if not normalized:
        raise ValueError("milestones must not be empty")
    if normalized[0] <= 0:
        raise ValueError("milestones must be positive integers")
    return normalized


def _normalize_runs(runs: list[int]) -> list[int]:
    normalized = sorted({int(run) for run in runs})
    if not normalized:
        raise ValueError("runs must not be empty")
    if normalized[0] <= 0:
        raise ValueError("runs must be positive integers")
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


def _harmonic_mean_ignore_none(values: list[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    if any(value <= 0 for value in valid_values):
        return 0.0
    return len(valid_values) / sum(1 / value for value in valid_values)


def _last_non_none(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _extract_score_series_from_records(
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


def _extract_value_series_from_records(
    records: list[dict[str, Any]],
    milestones: list[int],
    value_key: str,
) -> list[float | None]:
    value_by_step: dict[int, float | None] = {step: None for step in milestones}
    for record in records:
        step = record.get("step")
        if not isinstance(step, int) or step not in value_by_step:
            continue
        value = record.get(value_key)
        if value is not None:
            value_by_step[step] = float(value)

    return [value_by_step[step] for step in milestones]


def _compute_avg_time_series(total_time_series: list[float | None], milestones: list[int]) -> list[float | None]:
    return [
        None if total_time is None else total_time / step
        for total_time, step in zip(total_time_series, milestones)
    ]


def _method_directory(
    variant: str,
    model: str,
    method: str,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    return root / variant / model / method


def _result_path(
    variant: str,
    model: str,
    method: str,
    milestones: list[int],
    dataset_name: str,
    run_index: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    method_dir = _method_directory(variant, model, method, evaluation_dir)
    return method_dir / milestones_to_str(milestones) / f"{dataset_name}.{run_index}.json"


def _load_run_summary(
    variant: str,
    model: str,
    method: str,
    milestones: list[int],
    run_index: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    cfact_path = _result_path(variant, model, method, milestones, "cfact", run_index, evaluation_dir)
    mquake_path = _result_path(variant, model, method, milestones, "mquake", run_index, evaluation_dir)

    if not cfact_path.exists() or not mquake_path.exists():
        return None

    with open(cfact_path, encoding="utf-8") as file:
        cfact_records = json.load(file)
    with open(mquake_path, encoding="utf-8") as file:
        mquake_records = json.load(file)

    cfact_edit_time = _extract_value_series_from_records(cfact_records, milestones, EDIT_TIME_KEY)
    mquake_edit_time = _extract_value_series_from_records(mquake_records, milestones, EDIT_TIME_KEY)
    cfact_total_time = _extract_value_series_from_records(cfact_records, milestones, TOTAL_TIME_KEY)
    mquake_total_time = _extract_value_series_from_records(mquake_records, milestones, TOTAL_TIME_KEY)
    edit_time_series = [
        _mean_ignore_none([cfact_edit_time[idx], mquake_edit_time[idx]])
        for idx in range(len(milestones))
    ]
    total_time_series = [
        _mean_ignore_none([cfact_total_time[idx], mquake_total_time[idx]])
        for idx in range(len(milestones))
    ]
    avg_time_series = _compute_avg_time_series(total_time_series, milestones)

    return {
        "success": _extract_score_series_from_records(cfact_records, "success", milestones),
        "locality": _extract_score_series_from_records(cfact_records, "locality", milestones),
        "portability": _extract_score_series_from_records(mquake_records, "portability", milestones),
        "edit_time": edit_time_series,
        "total_time": total_time_series,
        "avg_time": avg_time_series,
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]], milestones: list[int]) -> dict[str, Any]:
    return {
        "success": [
            _mean_ignore_none([summary["success"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
        "locality": [
            _mean_ignore_none([summary["locality"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
        "portability": [
            _mean_ignore_none([summary["portability"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
        "edit_time": [
            _mean_ignore_none([summary["edit_time"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
        "total_time": [
            _mean_ignore_none([summary["total_time"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
        "avg_time": [
            _mean_ignore_none([summary["avg_time"][idx] for summary in run_summaries])
            for idx in range(len(milestones))
        ],
    }


def _load_results_with_skips(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    runs: list[int],
    evaluation_dir: Path | str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    milestones = _normalize_milestones(milestones)
    runs = _normalize_runs(runs)

    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    model_dir = root / variant / model
    if not model_dir.exists():
        return {}, {}

    data: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for method_name in methods:
        method_dir = model_dir / method_name
        if not method_dir.is_dir():
            skipped[method_name] = "missing method directory"
            continue

        run_summaries: list[dict[str, Any]] = []
        missing_runs: list[int] = []
        for run_index in runs:
            summary = _load_run_summary(
                variant=variant,
                model=model,
                method=method_name,
                milestones=milestones,
                run_index=run_index,
                evaluation_dir=root,
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
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    runs = _runs_from_count(n_runs)
    data, _ = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    return data


def _ordered_method_names(data: dict[str, Any], methods: list[str]) -> list[str]:
    return [method_name for method_name in methods if method_name in data]


def _load_performance_means(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, dict[str, float | None]]:
    data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        evaluation_dir=evaluation_dir,
    )
    return {
        method_name: {
            metric_name: _mean_ignore_none(summary[metric_name])
            for _, metric_name in METRIC_ORDER
        }
        for method_name, summary in data.items()
    }


def _quality_series(summary: dict[str, Any]) -> list[float | None]:
    return [
        _harmonic_mean_ignore_none([
            summary["success"][idx],
            summary["locality"][idx],
            summary["portability"][idx],
        ])
        for idx in range(len(summary["success"]))
    ]


def _composite_series(
    summary: dict[str, Any],
    time_ref: float,
    time_alpha: float,
) -> list[float | None]:
    if time_ref <= 0:
        raise ValueError("time_ref must be positive")
    if time_alpha < 0:
        raise ValueError("time_alpha must be non-negative")

    quality_series = _quality_series(summary)
    composite_values: list[float | None] = []
    for quality, edit_time in zip(quality_series, summary["edit_time"]):
        if quality is None or edit_time is None or edit_time <= 0:
            composite_values.append(None)
            continue
        time_score = min(time_ref / edit_time, 1.0)
        composite_values.append(quality * (time_score ** time_alpha))
    return composite_values


def _load_composite_means(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    time_ref: float,
    time_alpha: float,
    evaluation_dir: Path | str | None = None,
) -> dict[str, float | None]:
    data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        evaluation_dir=evaluation_dir,
    )
    return {
        method_name: _mean_ignore_none(_composite_series(summary, time_ref=time_ref, time_alpha=time_alpha))
        for method_name, summary in data.items()
    }


def _print_skipped(skipped: dict[str, str]) -> None:
    if skipped:
        details = ", ".join(f"{method} ({reason})" for method, reason in skipped.items())
        print(f"Skipped methods due to incomplete data: {details}")


def table_perf_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    mean_data = _load_performance_means(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        evaluation_dir=evaluation_dir,
    )
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
    print(f"performance | variant={variant} | model={model} | milestones={milestones} | n_runs={n_runs}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def table_time_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        time_series = data[method_name]["edit_time"]
        table_data[method_name] = [f"{value:.4f}" if value is not None else "-" for value in time_series]

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"milestone edit time | variant={variant} | model={model} | milestones={milestones} | n_runs={n_runs}")
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def table_composite_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    time_ref: float,
    time_alpha: float,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    mean_data = _load_composite_means(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        time_ref=time_ref,
        time_alpha=time_alpha,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones] + ["Mean"]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        composite_values = _composite_series(data[method_name], time_ref=time_ref, time_alpha=time_alpha)
        row = [f"{value:.4f}" if value is not None else "-" for value in composite_values]
        mean_value = mean_data[method_name]
        row.append(f"{mean_value:.4f}" if mean_value is not None else "-")
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(
        f"composite | variant={variant} | model={model} | milestones={milestones} | "
        f"n_runs={n_runs} | time_ref={time_ref} | time_alpha={time_alpha}"
    )
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


def plot_perf_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (14.0, 10.0),
) -> None:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    mean_data = _load_performance_means(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data or not mean_data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }

    fig, axes = plt.subplots(3, 2, figsize=figsize, squeeze=False)

    for metric_index, (short_name, metric_name) in enumerate(METRIC_ORDER):
        line_ax = axes[metric_index][0]
        bar_ax = axes[metric_index][1]
        has_any_line = False

        for method_name in methods_sorted:
            values = data[method_name][metric_name]
            if not any(value is not None for value in values):
                continue
            line_ax.plot(
                milestones,
                values,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=method_colors[method_name],
            )
            has_any_line = True

        if not has_any_line:
            line_ax.text(
                0.5, 0.5, "no data",
                ha="center", va="center",
                transform=line_ax.transAxes,
                fontsize=9,
                color="gray",
            )

        line_ax.set_ylim(-0.05, 1.05)
        line_ax.set_xticks(milestones)
        line_ax.set_xlabel("step")
        line_ax.set_ylabel("score")
        line_ax.set_title(f"{short_name} Sequential", fontweight="bold")
        line_ax.grid(alpha=0.25)

        values = [mean_data[method_name][metric_name] for method_name in methods_sorted]
        bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
        bar_values = [value for value in values if value is not None]
        bar_colors = [method_colors[method_name] for method_name in bar_methods]

        if bar_values:
            bar_ax.bar(bar_methods, bar_values, color=bar_colors, width=0.7)
        else:
            bar_ax.text(
                0.5, 0.5, "no data",
                ha="center", va="center",
                transform=bar_ax.transAxes,
                fontsize=9,
                color="gray",
            )
            bar_ax.set_xticks([])

        bar_ax.tick_params(axis="x", rotation=45)
        bar_ax.set_ylabel("mean score")
        bar_ax.set_ylim(-0.05, 1.05)
        bar_ax.set_title(f"{short_name} Mean", fontweight="bold")
        bar_ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=method_colors[method_name], lw=2, label=method_name)
        for method_name in methods_sorted
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="method")

    fig.suptitle(
        f"performance | variant={variant} model={model} milestones={milestones} n_runs={n_runs}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


def plot_time_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (12.0, 4.5),
) -> None:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    line_ax, bar_ax = axes[0]

    has_any_line = False
    for method_name in methods_sorted:
        values = data[method_name]["edit_time"]
        if not any(value is not None for value in values):
            continue
        line_ax.plot(
            milestones,
            values,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=method_colors[method_name],
        )
        has_any_line = True

    if not has_any_line:
        line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

    line_ax.set_xticks(milestones)
    line_ax.set_xlabel("step")
    line_ax.set_ylabel("edit time (s)")
    line_ax.set_title("Milestone Edit Time", fontweight="bold")
    line_ax.grid(alpha=0.25)

    final_times = [_last_non_none(data[method_name]["avg_time"]) for method_name in methods_sorted]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, final_times) if value is not None]
    bar_values = [value for value in final_times if value is not None]
    bar_colors = [method_colors[method_name] for method_name in bar_methods]

    if bar_values:
        bar_ax.bar(bar_methods, bar_values, color=bar_colors, width=0.7)
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])

    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("avg edit time (s)")
    bar_ax.set_title("Final Avg Time", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=method_colors[method_name], lw=2, label=method_name)
        for method_name in methods_sorted
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="method")

    fig.suptitle(
        f"time | variant={variant} model={model} milestones={milestones} n_runs={n_runs}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


def plot_composite_results(
    variant: str,
    model: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    time_ref: float,
    time_alpha: float,
    evaluation_dir: Path | str | None = None,
    figsize: tuple[float, float] = (12.0, 4.5),
) -> None:
    milestones = _normalize_milestones(milestones)
    runs = _runs_from_count(n_runs)
    data, skipped = _load_results_with_skips(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        runs=runs,
        evaluation_dir=evaluation_dir,
    )
    mean_data = _load_composite_means(
        variant=variant,
        model=model,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        time_ref=time_ref,
        time_alpha=time_alpha,
        evaluation_dir=evaluation_dir,
    )
    _print_skipped(skipped)
    if not data or not mean_data:
        print("No result files found for the selected settings.")
        return

    methods_sorted = _ordered_method_names(data, methods)
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    line_ax, bar_ax = axes[0]

    has_any_line = False
    for method_name in methods_sorted:
        values = _composite_series(data[method_name], time_ref=time_ref, time_alpha=time_alpha)
        if not any(value is not None for value in values):
            continue
        line_ax.plot(
            milestones,
            values,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=method_colors[method_name],
        )
        has_any_line = True

    if not has_any_line:
        line_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=line_ax.transAxes, fontsize=9, color="gray")

    line_ax.set_ylim(-0.05, 1.05)
    line_ax.set_xticks(milestones)
    line_ax.set_xlabel("step")
    line_ax.set_ylabel("composite score")
    line_ax.set_title("Composite Sequential", fontweight="bold")
    line_ax.grid(alpha=0.25)

    values = [mean_data[method_name] for method_name in methods_sorted]
    bar_methods = [method_name for method_name, value in zip(methods_sorted, values) if value is not None]
    bar_values = [value for value in values if value is not None]
    bar_colors = [method_colors[method_name] for method_name in bar_methods]

    if bar_values:
        bar_ax.bar(bar_methods, bar_values, color=bar_colors, width=0.7)
    else:
        bar_ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=bar_ax.transAxes, fontsize=9, color="gray")
        bar_ax.set_xticks([])

    bar_ax.tick_params(axis="x", rotation=45)
    bar_ax.set_ylabel("mean composite score")
    bar_ax.set_ylim(-0.05, 1.05)
    bar_ax.set_title("Composite Mean", fontweight="bold")
    bar_ax.grid(axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=method_colors[method_name], lw=2, label=method_name)
        for method_name in methods_sorted
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="method")

    fig.suptitle(
        f"composite | variant={variant} model={model} milestones={milestones} "
        f"n_runs={n_runs} time_ref={time_ref} time_alpha={time_alpha}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    plt.show()
