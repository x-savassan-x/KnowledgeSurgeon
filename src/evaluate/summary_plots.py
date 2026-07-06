from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .summary_utils import (
    METRIC_ORDER,
    _composite_series,
    _final_milestone_value,
    _final_values_and_stds,
    _load_composite_finals,
    _load_performance_finals,
    _load_results_with_skips,
    _normalize_milestones,
    _ordered_method_names,
    _performance_methods_with_base,
    _print_skipped,
    _quality_series,
    _series_for_metric,
    _std_series_for_metric,
    _valid_numbers,
    _zero_based_upper,
)


def _line_style_for_method(method_name: str) -> dict[str, Any]:
    if method_name == "base":
        return {"linestyle": "--", "linewidth": 1.9, "alpha": 0.9}
    return {"linestyle": "-", "linewidth": 1.6}


def _method_colors(methods_sorted: list[str]) -> dict[str, Any]:
    cmap = plt.colormaps.get_cmap("tab10")
    return {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }


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
