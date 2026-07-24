from pathlib import Path
from typing import Any

import pandas as pd

from .io import build_milestone_result_path, build_result_path, load_hardware_results
from .summary_utils import (
    METRIC_ORDER,
    _composite_series,
    _format_mean_std,
    _format_seconds_mean_std,
    _load_results_with_skips,
    _normalize_milestones,
    _ordered_method_names,
    _performance_methods_with_base,
    _print_skipped,
    _quality_series,
    _runs_from_count,
    _std_series_for_metric,
)


def _centered_style(df: pd.DataFrame, hide_index: bool = False) -> Any:
    style = df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    )
    if not hide_index:
        return style
    try:
        return style.hide(axis="index")
    except TypeError:
        return style.hide_index()


def table_perf_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    perf_methods = _performance_methods_with_base(
        model_key=model_key,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    data, skipped = _load_results_with_skips(
        model_key=model_key,
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
    print(f"performance | H=harmonic mean(ES, Loc, Port) | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


def table_time_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir)
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
    print(f"avg edit time | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


def table_composite_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        composite_values = _composite_series(data[method_name])
        composite_stds = _std_series_for_metric(data[method_name], "composite")
        table_data[method_name] = [
            _format_mean_std(value, std)
            for value, std in zip(composite_values, composite_stds)
        ]

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"composite | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


def table_memory_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    metrics = [
        ("allocated", "allocated_gb"),
        ("reserved", "reserved_gb"),
    ]
    columns = pd.MultiIndex.from_tuples([
        (f"step {step}", metric_label)
        for step in milestones
        for metric_label, _ in metrics
    ] + [
        ("overall peak", metric_label)
        for metric_label, _ in metrics
    ])

    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        summary = data[method_name]
        row: list[str] = []
        for step_index, _ in enumerate(milestones):
            for _, metric_name in metrics:
                row.append(_format_mean_std(
                    summary[metric_name][step_index],
                    summary.get(f"{metric_name}_std", [None] * len(milestones))[step_index],
                    decimals=2,
                    suffix=" GB",
                ))
        for _, metric_name in metrics:
            peak_name = f"peak_{metric_name}"
            row.append(_format_mean_std(
                summary.get(peak_name),
                summary.get(f"{peak_name}_std"),
                decimals=2,
                suffix=" GB",
            ))
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(
        table_data,
        orient="index",
        columns=columns,
    )
    df.index.name = "method"
    print(f"edit memory | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


def _edit_methods_only(methods: list[str]) -> list[str]:
    return [method_name for method_name in methods if method_name != "base"]


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


def _hardware_pcie_label(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "-"
    provider_label = _hardware_metadata_field(
        payload,
        (
            "provider_reported_pcie",
            "pcie_label",
            "pcie",
        ),
    )
    if provider_label != "-":
        return provider_label
    gpu_payload = payload.get("gpu", {})
    gpus = gpu_payload.get("gpus", []) if isinstance(gpu_payload, dict) else []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        gen = gpu.get("pcie_link_gen_current")
        width = gpu.get("pcie_link_width_current")
        if gen is not None and width is not None:
            return f"gen{gen} x{width}"
    return "-"


def _format_hardware_value(value, suffix: str = "") -> str:
    if value in (None, ""):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}{suffix}"


def _hardware_metadata_field(
    payload: dict[str, Any] | None,
    keys: tuple[str, ...],
    suffix: str = "",
) -> str:
    if not payload:
        return "-"
    for section_name in ("manual", "provider_env"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                return _format_hardware_value(section.get(key), suffix=suffix)
    return "-"


def table_hardware_results(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    rows: list[dict[str, Any]] = []
    for run_index in _runs_from_count(n_runs):
        payload = load_hardware_results(
            model_key=model_key,
            milestones=milestones,
            run_index=run_index,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        rows.append({
            "gpu": _hardware_gpu_label(payload),
            "gpu memory": _hardware_gpu_memory(payload),
            "gpu tflops": _hardware_metadata_field(
                payload,
                (
                    "provider_reported_tflops",
                    "gpu_tflops",
                    "tflops",
                    "fp16_tflops",
                    "gpu_fp16_tflops",
                ),
            ),
            "cpu": payload.get("cpu", {}).get("model", "-") if payload else "-",
            "cpu count": payload.get("cpu", {}).get("logical_count", "-") if payload else "-",
            "ram": (
                f"{payload.get('memory', {}).get('memtotal_gb'):.1f} GB"
                if payload and payload.get("memory", {}).get("memtotal_gb") is not None
                else "-"
            ),
            "pcie": _hardware_pcie_label(payload),
            "driver": _hardware_first_gpu_field(payload, "driver_version"),
            "dlperf": _hardware_metadata_field(
                payload,
                (
                    "provider_reported_dlperf",
                    "dlperf",
                ),
            ),
        })

    df = pd.DataFrame(rows)
    print(f"hardware | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df, hide_index=True)


def audit_result_grid(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    result_names: list[str] | None = None,
    result_groups: list[str] | None = None,
    evaluation_dir: Path | str | None = None,
    only_problems: bool = False,
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    result_names = result_names or ["es_loc", "port"]
    result_groups = result_groups or ["unscored", "scored", "perf", "time", "mem"]
    runs = _runs_from_count(n_runs)
    rows: list[dict[str, Any]] = []

    for method_name in methods:
        method_groups = ["perf"] if method_name == "base" else result_groups
        for result_group in method_groups:
            expected_paths: list[Path] = []
            for result_name in result_names:
                for run_index in runs:
                    if result_group in {"unscored", "scored"}:
                        expected_paths.extend(
                            build_milestone_result_path(
                                model_key=model_key,
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
                            for step in milestones
                        )
                    else:
                        expected_paths.append(
                            build_result_path(
                                model_key=model_key,
                                editing_method=method_name,
                                milestones=milestones,
                                result_name=result_name,
                                run_index=run_index,
                                n_runs=n_runs,
                                judge_repeats=judge_repeats,
                                result_group=result_group,
                                evaluation_dir=evaluation_dir,
                            )
                        )

            present = 0
            missing_files: list[str] = []
            for path in expected_paths:
                if path.exists() and path.stat().st_size > 0:
                    present += 1
                    continue
                suffix = " (empty)" if path.exists() else ""
                missing_files.append(f"{path.name}{suffix}")

            rows.append({
                "method": method_name,
                "group": result_group,
                "expected": len(expected_paths),
                "present": present,
                "missing": len(missing_files),
                "missing files": ", ".join(missing_files) if missing_files else "",
            })

    df = pd.DataFrame(rows)
    print(f"result-grid audit | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")

    if only_problems:
        problem_df = df[df["missing"] > 0].reset_index(drop=True)
        if problem_df.empty:
            print("No missing or empty files found for the selected grid.")
        return _centered_style(problem_df)
    return _centered_style(df)
