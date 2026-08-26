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
    _load_run_summary,
    _normalize_milestones,
    _ordered_method_names,
    _performance_methods_with_base,
    _print_skipped,
    _quality_series,
    _runs_from_count,
    _std_series_for_metric,
    load_sensitivity_runs,
)


def _intrinsic_value(summary: dict[str, Any] | None, key: str, index: int = -1) -> float | None:
    """Return a final intrinsic value while tolerating incomplete artifacts."""
    if not summary:
        return None
    values = summary.get(key)
    if not isinstance(values, list) or not values:
        return None
    value = values[index]
    return float(value) if value is not None else None


def table_intrinsic_results(
    study: str,
    intrinsic_config: dict[str, Any],
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    eval_type: str = "intrinsic",
) -> pd.DataFrame:
    """Render one compact table for the selected intrinsic study.

    Intrinsic run indices identify conditions rather than statistical repeats;
    consequently this function reports one row per configured condition.
    """
    studies = intrinsic_config.get("studies", {})
    study_config = studies.get(study)
    if not isinstance(study_config, dict):
        raise ValueError(f"Unknown intrinsic study: {study}")
    model_key = str(intrinsic_config["model_key"])
    n_runs = int(intrinsic_config.get("n_runs", len(study_config.get("runs", []))))
    milestones = [int(step) for step in study_config.get("eval_milestones", [])]
    if not milestones:
        raise ValueError(f"Intrinsic study {study!r} has no eval_milestones")
    milestone = milestones[-1]
    run_start = int(intrinsic_config.get("run_start", 1))
    run_count = intrinsic_config.get("run_count")
    run_stop = n_runs if run_count is None else min(n_runs, run_start + int(run_count) - 1)
    run_indices = range(run_start, run_stop + 1)

    rows: list[dict[str, Any]] = []
    runs = study_config.get("runs", [])
    for run_index in run_indices:
        run = runs[run_index - 1] if run_index - 1 < len(runs) else {"id": f"run{run_index}", "overrides": {}}
        condition_id = str(run.get("id", f"run{run_index}"))
        overrides = run.get("overrides", {}) if isinstance(run, dict) else {}
        if study == "intervention":
            strengths = study_config.get("strengths", study_config.get("intervention_strengths", []))
            for strength in strengths:
                strength_value = float(strength)
                slot = f"mnemo_s{int(round(strength_value * 100)):03d}"
                summary = _load_run_summary(
                    model_key, slot, milestones, run_index, n_runs, judge_repeats,
                    evaluation_dir, eval_type,
                )
                rows.append({
                    "max norm": overrides.get("delta", {}).get("max_norm", condition_id),
                    "strength": strength_value,
                    "ES": _intrinsic_value(summary, "success"),
                    "Loc": _intrinsic_value(summary, "locality"),
                    "Port": _intrinsic_value(summary, "portability"),
                    "time (s)": _intrinsic_value(summary, "avg_time"),
                })
        else:
            summary = _load_run_summary(
                model_key, "mnemo", milestones, run_index, n_runs, judge_repeats,
                evaluation_dir, eval_type,
            )
            label = condition_id
            if study == "encoder":
                label = overrides.get("router", {}).get("backend", condition_id)
                label = str(label)
            rows.append({
                "condition": label,
                "ES": _intrinsic_value(summary, "success"),
                "Loc": _intrinsic_value(summary, "locality"),
                "Port": _intrinsic_value(summary, "portability"),
                "time (s)": _intrinsic_value(summary, "avg_time"),
            })

    if not rows:
        print(f"No intrinsic result files found for study={study}.")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column in {"condition", "max norm"}:
            continue
        frame[column] = frame[column].map(
            lambda value: "-" if value is None or pd.isna(value) else f"{float(value):.2f}"
        )
    print(f"intrinsic {study} | model={model_key} | milestone={milestone} | eval_type={eval_type}")
    return _centered_style(frame, hide_index=True)


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
    show_std: bool = True,
    eval_type: str = "output",
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    perf_methods = _performance_methods_with_base(
        model_key=model_key,
        milestones=milestones,
        methods=methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
        eval_type=eval_type,
    )
    data, skipped = _load_results_with_skips(
        model_key=model_key,
        milestones=milestones,
        methods=perf_methods,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
        eval_type=eval_type,
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
                row.append(_format_mean_std(value, std, show_std=show_std))
            quality_value = _quality_series(summary)[step_index]
            quality_std = _std_series_for_metric(summary, "quality")[step_index]
            row.append(_format_mean_std(quality_value, quality_std, show_std=show_std))
        table_data[method_name] = row

    df = pd.DataFrame.from_dict(table_data, orient="index", columns=columns)
    df.index.name = "method"
    print(f"performance | H=harmonic mean(ES, Loc, Port) | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df)


def table_sensitivity_orders(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    eval_type: str = "sensitivity",
) -> pd.DataFrame:
    """Show each deterministic sensitivity order before aggregating runs."""
    milestones = _normalize_milestones(milestones)
    data, skipped = _load_results_with_skips(
        model_key, milestones, methods, n_runs, judge_repeats,
        evaluation_dir, eval_type,
    )
    _print_skipped(skipped)
    rows: list[dict[str, Any]] = []
    for method_name in _ordered_method_names(data, methods):
        for order_index, run in enumerate(data[method_name]["runs"], start=1):
            step = len(milestones) - 1
            rows.append({
                "method": method_name,
                "order": order_index,
                "ES": run["success"][step],
                "Loc": run["locality"][step],
                "Port": run["portability"][step],
                "time (s)": run["avg_time"][step],
            })
    if not rows:
        print("No sensitivity result files found for the selected settings.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for column in ("ES", "Loc", "Port", "time (s)"):
        df[column] = df[column].map(
            lambda value: "-" if value is None or pd.isna(value) else f"{float(value):.2f}"
        )
    print(f"sensitivity orders | model={model_key} | milestones={milestones} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df, hide_index=True)


def table_sensitivity_runs(
    model_keys: list[str],
    milestone: int,
    method: str,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    show_std: bool = False,
    eval_type: str = "sensitivity",
) -> pd.DataFrame:
    """Render matched sensitivity orders without collapsing them into repetitions."""
    rows = load_sensitivity_runs(
        model_keys=model_keys,
        milestone=milestone,
        method=method,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
        eval_type=eval_type,
    )
    if not rows:
        print("No sensitivity result files found for the selected settings.")
        return pd.DataFrame()
    columns = [
        "model", "order", "ES", "Loc", "Port",
        "ES/Loc avg edit time (s)", "Port avg edit time (s)",
    ]
    df = pd.DataFrame(rows)[columns]
    for column in columns[2:]:
        df[column] = df[column].map(lambda value: "-" if value is None else f"{float(value):.2f}")
    print(f"sensitivity runs | models={model_keys} | milestone={milestone} | method={method} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df, hide_index=True)


def table_sensitivity_variation(
    model_keys: list[str],
    milestone: int,
    method: str,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    show_std: bool = True,
    eval_type: str = "sensitivity",
) -> pd.DataFrame:
    """Summarize mean, standard deviation, and range across deterministic orders."""
    rows = load_sensitivity_runs(
        model_keys=model_keys,
        milestone=milestone,
        method=method,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
        eval_type=eval_type,
    )
    if not rows:
        print("No sensitivity result files found for the selected settings.")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    metric_columns = [
        "ES", "Loc", "Port",
        "ES/Loc avg edit time (s)", "Port avg edit time (s)",
    ]
    summary_rows: list[dict[str, Any]] = []
    for model, group in frame.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "orders": len(group)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column} mean"] = values.mean() if len(values) else None
            row[f"{column} std"] = values.std(ddof=1) if len(values) > 1 else 0.0 if len(values) else None
            row[f"{column} range"] = values.max() - values.min() if len(values) else None
        summary_rows.append(row)
    df = pd.DataFrame(summary_rows)
    display_columns = ["model", "orders"]
    for column in metric_columns:
        display_columns.append(f"{column} mean")
        if show_std:
            display_columns.append(f"{column} std")
        display_columns.append(f"{column} range")
    df = df[display_columns]
    for column in display_columns[2:]:
        df[column] = df[column].map(lambda value: "-" if value is None or pd.isna(value) else f"{float(value):.2f}")
    print(f"sensitivity variation | models={model_keys} | milestone={milestone} | method={method} | r{n_runs}-j{judge_repeats}")
    return _centered_style(df, hide_index=True)


def table_sensitivity_audit(
    model_keys: list[str],
    milestone: int,
    method: str,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    eval_type: str = "sensitivity",
) -> pd.DataFrame:
    """Check completeness, score recomputation, edit counts, and shared orders."""
    rows = load_sensitivity_runs(
        model_keys=model_keys,
        milestone=milestone,
        method=method,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
        eval_type=eval_type,
    )
    if not rows:
        print("No sensitivity result files found for the selected settings.")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    expected_es_order = frame.groupby("order")["ES/Loc order hash"].first().to_dict()
    expected_port_order = frame.groupby("order")["Port order hash"].first().to_dict()
    expected_es_cohort = frame["ES/Loc cohort hash"].dropna().iloc[0] if frame["ES/Loc cohort hash"].notna().any() else None
    expected_port_cohort = frame["Port cohort hash"].dropna().iloc[0] if frame["Port cohort hash"].notna().any() else None
    frame["complete"] = (
        (frame["ES/Loc edits"] == milestone)
        & (frame["Port edits"] == milestone)
        & (frame["ES details"] == milestone * 4)
        & (frame["Loc details"] == milestone * 4)
        & (frame["Port details"] == milestone * 3)
        & frame["aggregate valid"]
        & frame["time valid"]
        & frame["artifact files"]
        & frame.apply(lambda row: row["ES/Loc order hash"] == expected_es_order.get(row["order"]), axis=1)
        & frame.apply(lambda row: row["Port order hash"] == expected_port_order.get(row["order"]), axis=1)
        & frame["ES/Loc cohort hash"].eq(expected_es_cohort)
        & frame["Port cohort hash"].eq(expected_port_cohort)
    )
    df = frame[["model", "order", "ES/Loc edits", "Port edits", "ES details", "Loc details", "Port details", "aggregate valid", "time valid", "artifact files", "complete"]]
    print(f"sensitivity audit | models={model_keys} | milestone={milestone} | method={method} | complete={int(df['complete'].sum())}/{len(df)}")
    return _centered_style(df, hide_index=True)


def table_time_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    show_std: bool = True,
    eval_type: str = "output",
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir, eval_type)
    _print_skipped(skipped)
    if not data:
        print("No result files found for the selected settings.")
        return pd.DataFrame()

    columns = [f"step {step}" for step in milestones]
    table_data: dict[str, list[str]] = {}
    for method_name in _ordered_method_names(data, methods):
        summary = data[method_name]
        table_data[method_name] = [
            _format_seconds_mean_std(value, std, show_std=show_std)
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
    show_std: bool = True,
    eval_type: str = "output",
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir, eval_type)
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
            _format_mean_std(value, std, show_std=show_std)
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
    show_std: bool = True,
    eval_type: str = "output",
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    methods = _edit_methods_only(methods)
    data, skipped = _load_results_with_skips(model_key, milestones, methods, n_runs, judge_repeats, evaluation_dir, eval_type)
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
                    show_std=show_std,
                ))
        for _, metric_name in metrics:
            peak_name = f"peak_{metric_name}"
            row.append(_format_mean_std(
                summary.get(peak_name),
                summary.get(f"{peak_name}_std"),
                decimals=2,
                suffix=" GB",
                show_std=show_std,
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
    gpu_payload = payload.get("gpu", {})
    gpus = gpu_payload.get("gpus", []) if isinstance(gpu_payload, dict) else []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        gen = gpu.get("pcie_link_gen_max")
        width = gpu.get("pcie_link_width_max")
        if gen is None or width is None:
            gen = gpu.get("pcie_link_gen_current")
            width = gpu.get("pcie_link_width_current")
        if gen is not None and width is not None:
            return f"gen{gen} x{width}"
    return "-"


def _hardware_cpu_topology(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "-"
    cpu = payload.get("cpu", {})
    if not isinstance(cpu, dict):
        return "-"
    parts = []
    for key, suffix in (
        ("socket_count", "S"),
        ("physical_core_count", "C"),
        ("logical_count", "T"),
        ("numa_node_count", "N"),
    ):
        value = cpu.get(key)
        if value is not None:
            parts.append(f"{value}{suffix}")
    if cpu.get("l3_cache"):
        parts.append(f"L3 {cpu['l3_cache']}")
    return "/".join(parts) if parts else "-"


def _hardware_cpu_allocation(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "-"
    cpu = payload.get("cpu", {})
    if not isinstance(cpu, dict):
        return "-"
    effective = cpu.get("effective_logical_count")
    quota = cpu.get("quota_cores")
    if effective is None and quota is None:
        return "-"
    if quota is None:
        return f"{effective} affinity"
    if effective is None:
        return f"quota {float(quota):g}"
    return f"{effective} affinity / {float(quota):g} quota"


def _hardware_gpu_power(payload: dict[str, Any] | None) -> str:
    value = _hardware_first_gpu_field(payload, "power_limit_w")
    if value == "-":
        return value
    try:
        return f"{float(value):g} W"
    except ValueError:
        return value


def table_hardware_results(
    model_key: str,
    milestones: list[int],
    methods: list[str],
    result_names: list[str],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
    eval_type: str = "output",
) -> pd.DataFrame:
    milestones = _normalize_milestones(milestones)
    rows: list[dict[str, Any]] = []
    for method_name in _edit_methods_only(methods):
        for result_name in result_names:
            for run_index in _runs_from_count(n_runs):
                payload = load_hardware_results(
                    model_key=model_key,
                    editing_method=method_name,
                    milestones=milestones,
                    result_name=result_name,
                    run_index=run_index,
                    n_runs=n_runs,
                    judge_repeats=judge_repeats,
                    evaluation_dir=evaluation_dir,
                    eval_type=eval_type,
                )
                if payload is None:
                    continue
                rows.append({
                    "method": method_name,
                    "result type": result_name,
                    "run": run_index,
                    "gpu power": _hardware_gpu_power(payload),
                    "pcie": _hardware_pcie_label(payload),
                    "cpu": payload.get("cpu", {}).get("model", "-"),
                    "cpu topology": _hardware_cpu_topology(payload),
                    "cpu allocation": _hardware_cpu_allocation(payload),
                    "ram": (
                        f"{payload.get('memory', {}).get('memtotal_gb'):.1f} GB"
                        if payload.get("memory", {}).get("memtotal_gb") is not None
                        else "-"
                    ),
                    "driver": _hardware_first_gpu_field(payload, "driver_version"),
                })

    df = pd.DataFrame(rows)
    print(
        f"hardware | model={model_key} | milestones={milestones} | "
        f"r{n_runs}-j{judge_repeats} | files={len(rows)}"
    )
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
    eval_type: str = "output",
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
                                eval_type=eval_type,
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
                                eval_type=eval_type,
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
