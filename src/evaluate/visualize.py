import json
from pathlib import Path
from typing import Any
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from .io import EVALUATION_DIR, milestones_to_str


METRIC_ORDER: list[tuple[str, str]] = [
    ("ES", "success"),
    ("Loc", "locality"),
    ("Port", "portability"),
]

TIME_KEY = "avg_edit_time"


def _normalize_filter(values: str | list[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return set(values)


def _normalize_milestones(milestones: list[int]) -> list[int]:
    normalized = sorted({int(step) for step in milestones})
    if not normalized:
        raise ValueError("milestones must not be empty")
    if normalized[0] <= 0:
        raise ValueError("milestones must be positive integers")
    return normalized


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


def _extract_avg_edit_time(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None

    final_record = max(
        (record for record in records if isinstance(record.get("step"), int)),
        key=lambda record: record["step"],
        default=None,
    )
    if final_record is None:
        return None

    run_stats = final_record.get("run_stats")
    if not isinstance(run_stats, dict):
        return None

    avg_edit_time = run_stats.get(TIME_KEY)
    if avg_edit_time is None:
        return None
    return float(avg_edit_time)


def _method_directory(
    variant: str,
    model: str,
    method: str,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    return root / variant / model / method


def _load_method_metrics(
    variant: str,
    model: str,
    method: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
) -> dict[str, list[float | None]] | None:
    milestones_dir = milestones_to_str(milestones)
    method_dir = _method_directory(variant, model, method, evaluation_dir)
    cfact_path = method_dir / milestones_dir / "cfact.json"
    mquake_path = method_dir / milestones_dir / "mquake.json"

    # Only include complete methods for this milestone configuration.
    if not cfact_path.exists() or not mquake_path.exists():
        return None

    # Plain json loads are intentionally used to preserve the pipeline schema.
    with open(cfact_path, encoding="utf-8") as file:
        cfact_records = json.load(file)
    with open(mquake_path, encoding="utf-8") as file:
        mquake_records = json.load(file)

    time_values = [
        value
        for value in (
            _extract_avg_edit_time(cfact_records),
            _extract_avg_edit_time(mquake_records),
        )
        if value is not None
    ]

    return {
        "success": _extract_score_series_from_records(cfact_records, "success", milestones),
        "locality": _extract_score_series_from_records(cfact_records, "locality", milestones),
        "portability": _extract_score_series_from_records(mquake_records, "portability", milestones),
        TIME_KEY: mean(time_values) if time_values else None,
    }


def load_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
) -> dict[str, dict[str, list[float | None]]]:
    """Load scores keyed by method for one variant/model and milestone setting."""
    milestones = _normalize_milestones(milestones)
    method_filter = _normalize_filter(methods)

    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    model_dir = root / variant / model
    data: dict[str, dict[str, list[float | None]]] = {}

    if not model_dir.exists():
        return data

    for method_dir in sorted(model_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        if method_filter and method_name not in method_filter:
            continue

        metrics = _load_method_metrics(
            variant=variant,
            model=model,
            method=method_name,
            milestones=milestones,
            evaluation_dir=root,
        )
        if metrics is None:
            continue
        data[method_name] = metrics

    return data


def load_mean_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Load one mean value per metric for each method across the selected milestones."""
    data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )

    return {
        method_name: {
            metric_name: _mean_ignore_none(metrics[metric_name])
            for _, metric_name in METRIC_ORDER
        }
        for method_name, metrics in data.items()
    }


def table_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
) -> pd.DataFrame:
    """Display a table where rows are methods and columns are milestone x {ES, Loc, Port}."""
    milestones = _normalize_milestones(milestones)
    data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    if not data:
        print("No result files found for the selected variant/model/milestones.")
        return pd.DataFrame()

    columns = pd.MultiIndex.from_tuples(
        [
            (f"step {step}", metric_short)
            for step in milestones
            for metric_short, _ in METRIC_ORDER
        ]
    )

    table_data: dict[str, list[str]] = {}
    for method_name in sorted(data.keys()):
        metrics = data[method_name]
        row: list[str] = []
        for step_index, _ in enumerate(milestones):
            for _, metric_name in METRIC_ORDER:
                value = metrics[metric_name][step_index]
                row.append(f"{value:.4f}" if value is not None else "-")
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"variant={variant} | model={model} | milestones={milestones}")
    # display(df)
    styled_df = df.style.set_properties(**{'text-align': 'center'}) \
                        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    return styled_df


def table_mean_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
) -> pd.DataFrame:
    """Display one mean value per metric for each method across the selected milestones, plus average edit time."""
    data = load_mean_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    full_data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    if not data or not full_data:
        print("No result files found for the selected variant/model/milestones.")
        return pd.DataFrame()

    columns = [short_name for short_name, _ in METRIC_ORDER] + ["Time"]
    table_data: dict[str, list[str]] = {}
    for method_name in sorted(data.keys()):
        row = []
        for _, metric_name in METRIC_ORDER:
            value = data[method_name][metric_name]
            row.append(f"{value:.4f}" if value is not None else "-")
        avg_edit_time = full_data[method_name][TIME_KEY]
        row.append(f"{avg_edit_time:.4f}" if avg_edit_time is not None else "-")
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"metric means | variant={variant} | model={model} | milestones={milestones}")
    styled_df = df.style.set_properties(**{'text-align': 'center'}) \
                        .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    return styled_df


def plot_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
    figsize: tuple[float, float] = (15.0, 4.0),
) -> None:
    """Draw three side-by-side line charts: ES, Loc, and Port."""
    milestones = _normalize_milestones(milestones)
    data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    if not data:
        print("No result files found for the selected variant/model/milestones.")
        return

    methods_sorted = sorted(data.keys())
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }

    fig, axes = plt.subplots(1, 3, figsize=figsize, squeeze=False)
    axes_row = axes[0]

    for metric_index, (short_name, metric_name) in enumerate(METRIC_ORDER):
        ax = axes_row[metric_index]
        has_any_line = False

        for method_name in methods_sorted:
            values = data[method_name][metric_name]
            if not any(value is not None for value in values):
                continue
            ax.plot(
                milestones,
                values,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=method_colors[method_name],
                label=method_name,
            )
            has_any_line = True

        if not has_any_line:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(milestones)
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.set_title(short_name, fontweight="bold")
        ax.grid(alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=method_colors[method_name], lw=2, label=method_name)
        for method_name in methods_sorted
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), title="method")

    fig.suptitle(f"variant={variant} model={model} milestones={milestones}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_mean_results(
    variant: str,
    model: str,
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    methods: str | list[str] | None = None,
    figsize: tuple[float, float] = (20.0, 4.5),
) -> None:
    """Draw one bar chart per metric, plus one for average edit time."""
    mean_data = load_mean_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    full_data = load_results(
        variant=variant,
        model=model,
        milestones=milestones,
        evaluation_dir=evaluation_dir,
        methods=methods,
    )
    if not mean_data or not full_data:
        print("No result files found for the selected variant/model/milestones.")
        return

    methods_sorted = sorted(mean_data.keys())
    panel_specs = list(METRIC_ORDER) + [("Time", TIME_KEY)]
    cmap = plt.colormaps.get_cmap("tab10")
    method_colors = {
        method_name: cmap(index % max(1, len(cmap.colors)))
        for index, method_name in enumerate(methods_sorted)
    }

    fig, axes = plt.subplots(1, len(panel_specs), figsize=figsize, squeeze=False)
    axes_row = axes[0]

    for idx, (label, metric_name) in enumerate(panel_specs):
        ax = axes_row[idx]
        if metric_name == TIME_KEY:
            values = [full_data[method_name][TIME_KEY] for method_name in methods_sorted]
            ylabel = "seconds"
            title = "Time"
            ylim = None
        else:
            values = [mean_data[method_name][metric_name] for method_name in methods_sorted]
            ylabel = "mean score"
            title = label
            ylim = (-0.05, 1.05)

        bar_methods = [
            method_name
            for method_name, value in zip(methods_sorted, values)
            if value is not None
        ]
        bar_values = [value for value in values if value is not None]
        bar_colors = [method_colors[method_name] for method_name in bar_methods]

        if bar_values:
            ax.bar(
                bar_methods,
                bar_values,
                color=bar_colors,
                width=0.7,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )
            ax.set_xticks([])

        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle(f"variant={variant} model={model} milestones={milestones}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()
