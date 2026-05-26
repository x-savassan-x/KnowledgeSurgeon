import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from IPython.display import display

from .io import EVALUATION_DIR, iter_result_files, load_result_file, milestones_to_str


METRIC_ORDER: list[tuple[str, str]] = [
    ("ES", "success"),
    ("Loc", "locality"),
    ("Port", "portability"),
]


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


def catalog_results(
    evaluation_dir: Path | str | None = None,
    variant: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Return one row per saved result file."""
    rows: list[dict[str, Any]] = []
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    variant_filter = _normalize_filter(variant)
    model_filter = _normalize_filter(model)

    for path in iter_result_files(root):
        loaded = load_result_file(path, root)
        if loaded is None:
            continue

        meta, records = loaded
        run_variant = str(meta["variant"])
        run_model = str(meta["model"])

        if variant_filter and run_variant not in variant_filter:
            continue
        if model_filter and run_model not in model_filter:
            continue

        rows.append(
            {
                "variant": run_variant,
                "model": run_model,
                "method": meta["method"],
                "milestones": milestones_to_str(meta["milestones"]),
                "dataset_name": meta["dataset_name"],
                "num_records": len(records),
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "variant",
                "model",
                "method",
                "milestones",
                "dataset_name",
                "num_records",
                "path",
            ]
        )

    return pd.DataFrame(rows).sort_values(["variant", "model", "milestones", "method", "dataset_name"])


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

    return {
        "success": _extract_score_series_from_records(cfact_records, "success", milestones),
        "locality": _extract_score_series_from_records(cfact_records, "locality", milestones),
        "portability": _extract_score_series_from_records(mquake_records, "portability", milestones),
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
