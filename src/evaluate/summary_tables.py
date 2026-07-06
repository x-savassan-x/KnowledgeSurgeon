from pathlib import Path
from typing import Any

import pandas as pd

from .io import build_hardware_path, build_milestone_result_path, build_result_path, load_hardware_results
from .summary_utils import (
    METRIC_ORDER,
    _composite_series,
    _final_milestone_value,
    _format_mean_std,
    _format_seconds_mean_std,
    _load_composite_finals,
    _load_results_with_skips,
    _normalize_milestones,
    _ordered_method_names,
    _performance_methods_with_base,
    _print_skipped,
    _quality_series,
    _runs_from_count,
    _std_series_for_metric,
)


def _centered_style(df: pd.DataFrame):
    return df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )


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
    return _centered_style(df)


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
    return _centered_style(df)


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
    return _centered_style(df)


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
    return _centered_style(df)


def _hardware_gpu_label(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "-"
    gpu_payload = payload.get("gpu", {})
    gpus = gpu_payload.get("gpus", []) if isinstance(gpu_payload, dict) else []
    if not gpus:
        torch_devices = payload.get("torch", {}).get("devices", [])
        gpus = torch_devices if isinstance(torch_devices, list) else []
    if not gpus:
        return "-"
    names = [str(gpu.get("name", "unknown")) for gpu in gpus if isinstance(gpu, dict)]
    unique_names = list(dict.fromkeys(names))
    if len(unique_names) == 1:
        return f"{len(gpus)}x {unique_names[0]}"
    return ", ".join(unique_names)


def _hardware_gpu_memory(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "-"
    gpu_payload = payload.get("gpu", {})
    gpus = gpu_payload.get("gpus", []) if isinstance(gpu_payload, dict) else []
    values = [
        float(gpu["memory_total_gb"])
        for gpu in gpus
        if isinstance(gpu, dict) and gpu.get("memory_total_gb") is not None
    ]
    if not values:
        values = [
            float(gpu["total_memory_gb"])
            for gpu in payload.get("torch", {}).get("devices", [])
            if isinstance(gpu, dict) and gpu.get("total_memory_gb") is not None
        ]
    if not values:
        return "-"
    return f"{sum(values):.1f} GB"


def _hardware_first_gpu_field(payload: dict[str, Any] | None, field: str) -> str:
    if not payload:
        return "-"
    gpu_payload = payload.get("gpu", {})
    gpus = gpu_payload.get("gpus", []) if isinstance(gpu_payload, dict) else []
    for gpu in gpus:
        if isinstance(gpu, dict) and gpu.get(field) is not None:
            return str(gpu[field])
    return "-"


def table_hardware_results(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    rows: list[dict[str, Any]] = []
    for run_index in _runs_from_count(n_runs):
        payload = load_hardware_results(
            edited_model=model,
            milestones=milestones,
            run_index=run_index,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        path = build_hardware_path(
            edited_model=model,
            milestones=milestones,
            run_index=run_index,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        rows.append({
            "run": run_index,
            "status": "present" if payload is not None else "missing",
            "captured_at": payload.get("captured_at", "-") if payload else "-",
            "hostname": payload.get("hostname", "-") if payload else "-",
            "gpu": _hardware_gpu_label(payload),
            "gpu memory": _hardware_gpu_memory(payload),
            "driver": _hardware_first_gpu_field(payload, "driver_version"),
            "torch cuda": payload.get("torch", {}).get("cuda_version", "-") if payload else "-",
            "cpu": payload.get("cpu", {}).get("model", "-") if payload else "-",
            "cpu count": payload.get("cpu", {}).get("logical_count", "-") if payload else "-",
            "ram": (
                f"{payload.get('memory', {}).get('memtotal_gb'):.1f} GB"
                if payload and payload.get("memory", {}).get("memtotal_gb") is not None
                else "-"
            ),
            "path": str(path),
        })

    df = pd.DataFrame(rows)
    print(f"hardware | model={model} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


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
