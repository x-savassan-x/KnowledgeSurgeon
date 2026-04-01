import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


EVALUATION_DIR = Path(__file__).parent.parent / "data" / "evaluation"
METRICS = ["success", "locality", "portability"]
FILE_PATTERN = re.compile(
    r"^(?P<model>.+)_(?P<dataset>cfact|mquake)_(?P<milestones>\d+(?:-\d+)*)\.json$"
)


def load_results(
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
) -> dict[tuple[str, str], dict[str, list[float | None]]]:
    """Load evaluation files matching the given milestones.

    Returns
    -------
    {(method, model): {"success": [score_m1, ...],
                       "locality": [...],
                       "portability": [...]}}
    """
    milestones = sorted(milestones)
    if not milestones:
        raise ValueError("milestones must not be empty")

    milestone_str = "-".join(str(milestone) for milestone in milestones)
    step_to_index = {step: index for index, step in enumerate(milestones)}
    evaluation_dir = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR

    data: dict[tuple[str, str], dict[str, list[float | None]]] = {}

    if not evaluation_dir.exists():
        return data

    for method_dir in sorted(evaluation_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name

        for fpath in sorted(method_dir.iterdir()):
            match = FILE_PATTERN.match(fpath.name)
            if not match:
                continue
            if match["milestones"] != milestone_str:
                continue

            dataset = match["dataset"]
            model = match["model"]
            key = (method, model)

            if key not in data:
                data[key] = {metric: [None] * len(milestones) for metric in METRICS}

            with open(fpath, encoding="utf-8") as file:
                records = json.load(file)

            for record in records:
                step = record.get("step")
                if step not in step_to_index:
                    continue
                index = step_to_index[step]

                result = record.get("result", record)
                if dataset == "cfact":
                    success = result.get("success")
                    locality = result.get("locality")
                    if success is not None:
                        data[key]["success"][index] = success.get("score")
                    if locality is not None:
                        data[key]["locality"][index] = locality.get("score")
                elif dataset == "mquake":
                    portability = result.get("portability")
                    if portability is not None:
                        data[key]["portability"][index] = portability.get("score")

    return data


def _collect_axes(data: dict[tuple[str, str], dict[str, list[float | None]]]) -> tuple[list[str], list[str]]:
    methods = sorted({method for method, _ in data})
    models = sorted({model for _, model in data})
    return methods, models


def plot_results(
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
    figsize_per_cell: tuple[float, float] = (4.5, 3.5),
) -> None:
    """Draw a grid of line graphs for success, locality, and portability."""
    milestones = sorted(milestones)
    data = load_results(milestones, evaluation_dir)
    if not data:
        print("No result files found.")
        return

    methods, models = _collect_axes(data)
    n_rows, n_cols = len(models), len(methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    x = milestones

    for row_index, model in enumerate(models):
        for col_index, method in enumerate(methods):
            ax = axes[row_index][col_index]
            key = (method, model)
            scores = data.get(key)

            if scores is None:
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
            else:
                for metric in METRICS:
                    values = scores[metric]
                    if any(value is not None for value in values):
                        ax.plot(x, values, marker="o", markersize=4, label=metric)
                ax.legend(fontsize=7)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(x)
            ax.set_xlabel("step")
            ax.set_ylabel("score")

            if row_index == 0:
                ax.set_title(method, fontweight="bold")
            if col_index == 0:
                ax.annotate(
                    model,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                )

    fig.suptitle(
        f"milestones={milestones}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


def table_results(
    milestones: list[int],
    evaluation_dir: Path | str | None = None,
) -> None:
    """Display tables where rows are metrics and columns are milestone steps."""
    milestones = sorted(milestones)
    data = load_results(milestones, evaluation_dir)
    if not data:
        print("No result files found.")
        return

    methods, models = _collect_axes(data)
    step_columns = [f"step {step}" for step in milestones]

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model}")
        print(f"{'=' * 60}")
        for method in methods:
            key = (method, model)
            scores = data.get(key)

            print(f"\n  Method: {method}")
            if scores is None:
                print("    (no data)")
                continue

            rows: dict[str, list[str]] = {}
            for metric in METRICS:
                values = scores[metric]
                rows[metric] = [f"{value:.4f}" if value is not None else "—" for value in values]

            df = pd.DataFrame(rows, index=step_columns).T
            df.index.name = "metric"
            display(df)
