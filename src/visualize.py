import json
import re

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
METRICS = ["reliability", "locality", "portability"]
FILE_PATTERN = re.compile(
    r"^(?P<dataset>cfact|mquake)_(?P<model>.+)_(?P<steps>\d+)_(?P<nonseq>\d+)\.json$"
)


def load_results(
    steps: int,
    non_seq_per_step: int,
    results_dir: Path | str | None = None,
) -> dict[tuple[str, str], dict[str, list[float | None]]]:
    """Load result files matching the given steps and non_seq_per_step.

    Returns
    -------
    {(method, model): {"reliability": [score_step1, ...],
                       "locality": [...],
                       "portability": [...]}}
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    data: dict[tuple[str, str], dict[str, list[float | None]]] = {}

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name

        for fpath in sorted(method_dir.iterdir()):
            m = FILE_PATTERN.match(fpath.name)
            if not m:
                continue
            if int(m["steps"]) != steps or int(m["nonseq"]) != non_seq_per_step:
                continue

            dataset = m["dataset"]
            model = m["model"]
            key = (method, model)

            if key not in data:
                data[key] = {metric: [None] * steps for metric in METRICS}

            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)

            for rec in records:
                idx = rec["step"] - 1
                if idx < 0 or idx >= steps:
                    continue
                if dataset == "cfact":
                    if "reliability" in rec and rec["reliability"] is not None:
                        data[key]["reliability"][idx] = rec["reliability"]
                    if "locality" in rec and rec["locality"] is not None:
                        data[key]["locality"][idx] = rec["locality"]
                elif dataset == "mquake":
                    if "portability" in rec and rec["portability"] is not None:
                        data[key]["portability"][idx] = rec["portability"]

    return data


def _collect_axes(data):
    """Return sorted lists of methods and models present in the data."""
    methods = sorted({m for m, _ in data})
    models = sorted({mod for _, mod in data})
    return methods, models


def plot_results(
    steps: int,
    non_seq_per_step: int,
    results_dir: Path | str | None = None,
    figsize_per_cell: tuple[float, float] = (4.5, 3.5),
):
    """Draw a grid of line graphs.

    Rows = models (horizontal lines of graphs),
    Columns = editing methods.
    Each cell plots reliability, locality, portability across steps.
    """
    data = load_results(steps, non_seq_per_step, results_dir)
    if not data:
        print("No result files found.")
        return

    methods, models = _collect_axes(data)
    n_rows, n_cols = len(models), len(methods)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    x = list(range(1, steps + 1))

    for ri, model in enumerate(models):
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]
            key = (method, model)
            scores = data.get(key)

            if scores is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                for metric in METRICS:
                    vals = scores[metric]
                    if any(v is not None for v in vals):
                        ax.plot(x, vals, marker="o", markersize=4, label=metric)

                ax.legend(fontsize=7)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(x)
            ax.set_xlabel("step")
            ax.set_ylabel("score")

            if ri == 0:
                ax.set_title(method, fontweight="bold")
            if ci == 0:
                ax.annotate(
                    model, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords="offset points",
                    ha="right", va="center", fontweight="bold", fontsize=10,
                )

    fig.suptitle(
        f"steps={steps}, non_seq_per_step={non_seq_per_step}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.show()


def table_results(
    steps: int,
    non_seq_per_step: int,
    results_dir: Path | str | None = None,
):
    """Display a grid of tables.

    Rows = models, Columns = editing methods.
    Each table: rows = metrics (reliability, locality, portability),
                columns = step numbers.
    """
    data = load_results(steps, non_seq_per_step, results_dir)
    if not data:
        print("No result files found.")
        return

    methods, models = _collect_axes(data)

    step_cols = [f"step {i+1}" for i in range(steps)]

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")
        for method in methods:
            key = (method, model)
            scores = data.get(key)

            print(f"\n  Method: {method}")
            if scores is None:
                print("    (no data)")
                continue

            rows = {}
            for metric in METRICS:
                vals = scores[metric]
                rows[metric] = [f"{v:.4f}" if v is not None else "—" for v in vals]

            df = pd.DataFrame(rows, index=step_cols).T
            df.index.name = "metric"
            display(df)
