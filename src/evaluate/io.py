import json
import os
import re
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"
RESULT_GROUPS = {"perf", "unscored", "scored", "time", "mem", "hardware"}


def _normalize_milestones(milestones: list[int]) -> list[int]:
    if not milestones:
        raise ValueError("milestones must not be empty")

    normalized = sorted({int(step) for step in milestones})
    if normalized[0] <= 0:
        raise ValueError("milestones must be positive integers")

    return normalized


def _validate_run_settings(n_runs: int, judge_repeats: int) -> None:
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")
    if judge_repeats <= 0:
        raise ValueError("judge_repeats must be > 0")


def milestones_to_str(milestones: list[int]) -> str:
    return "-".join(str(step) for step in _normalize_milestones(milestones))


def run_setting_to_str(n_runs: int, judge_repeats: int) -> str:
    _validate_run_settings(n_runs, judge_repeats)
    return f"r{n_runs}-j{judge_repeats}"


def build_method_dir(
    model_key: str,
    editing_method: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    return build_experiment_dir(
        model_key=model_key,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    ) / editing_method


def build_experiment_dir(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    return (
        root
        / model_key
        / milestones_to_str(milestones)
        / run_setting_to_str(n_runs, judge_repeats)
    )


def build_hardware_path(
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    if run_index <= 0:
        raise ValueError("run_index must be > 0")
    if not result_name:
        raise ValueError("result_name must not be empty")
    return (
        build_method_dir(
            model_key=model_key,
            editing_method=editing_method,
            milestones=milestones,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        / "hardware"
        / f"{result_name}_{run_index}.json"
    )


def build_result_path(
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    result_group: str = "perf",
    evaluation_dir: Path | str | None = None,
) -> Path:
    if run_index <= 0:
        raise ValueError("run_index must be > 0")
    standard_groups = RESULT_GROUPS - {"hardware"}
    if result_group not in standard_groups:
        raise ValueError(f"result_group must be one of: {', '.join(sorted(standard_groups))}")
    method_dir = build_method_dir(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    return method_dir / result_group / f"{result_name}.{run_index}.json"


def build_milestone_result_path(
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    step: int,
    n_runs: int,
    judge_repeats: int,
    result_group: str,
    evaluation_dir: Path | str | None = None,
) -> Path:
    if run_index <= 0:
        raise ValueError("run_index must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if result_group not in {"unscored", "scored"}:
        raise ValueError("result_group must be one of: unscored, scored")
    method_dir = build_method_dir(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    return method_dir / result_group / f"{result_name}.{run_index}.{step}.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return

    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def save_benchmark_results(
    results: list[dict[str, Any]],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_result_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="perf",
        evaluation_dir=evaluation_dir,
    )
    _write_json(path, results)

    return path


def save_unscored_results(
    result: dict[str, Any],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    step: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_milestone_result_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        step=step,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="unscored",
        evaluation_dir=evaluation_dir,
    )

    _write_json(path, result)

    return path


def save_scored_results(
    result: dict[str, Any],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    step: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_milestone_result_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        step=step,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="scored",
        evaluation_dir=evaluation_dir,
    )

    _write_json(path, result)

    return path


def save_time_results(
    edit_times: list[float],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_result_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="time",
        evaluation_dir=evaluation_dir,
    )
    payload = {"edit": edit_times}
    _write_json(path, payload)

    return path


def save_memory_results(
    memory: dict[str, Any],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_result_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="mem",
        evaluation_dir=evaluation_dir,
    )
    _write_json(path, memory)

    return path


def save_hardware_results(
    hardware: dict[str, Any],
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_hardware_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    hardware = dict(hardware)
    for key in (
        "schema_version",
        "run_index",
        "captured_at",
        "notes",
        "provider_env",
    ):
        hardware.pop(key, None)
    _write_json(path, hardware)

    return path


def load_hardware_results(
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    path = build_hardware_path(
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def iter_result_files(evaluation_dir: Path | str | None = None) -> list[Path]:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_result_file(
    path: Path,
    evaluation_dir: Path | str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | dict[str, Any]] | None:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    root = root.resolve()
    path = path.resolve()

    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) != 6:
        return None

    model_key, milestones_str, run_setting, editing_method, result_group, filename = parts
    if not filename.endswith(".json"):
        return None
    if result_group not in RESULT_GROUPS:
        return None

    try:
        milestones = [int(step) for step in milestones_str.split("-")]
        milestones = _normalize_milestones(milestones)
    except ValueError:
        return None

    match = re.fullmatch(r"r(\d+)-j(\d+)", run_setting)
    if match is None:
        return None
    n_runs = int(match.group(1))
    judge_repeats = int(match.group(2))

    stem = filename[:-5]
    step = None
    if result_group in {"unscored", "scored"}:
        parts = stem.rsplit(".", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            return None
        result_name = parts[0]
        run_index = int(parts[1])
        step = int(parts[2])
    elif result_group == "hardware":
        match = re.fullmatch(r"(.+)_(\d+)", stem)
        if match is None:
            return None
        result_name = match.group(1)
        run_index = int(match.group(2))
    else:
        result_name = stem
        run_index = None
        if "." in stem:
            maybe_result_name, maybe_run_index = stem.rsplit(".", 1)
            if maybe_run_index.isdigit():
                result_name = maybe_result_name
                run_index = int(maybe_run_index)
    if not result_name:
        return None

    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    meta = {
        "model_key": model_key,
        "method": editing_method,
        "milestones": milestones,
        "n_runs": n_runs,
        "judge_repeats": judge_repeats,
        "result_group": result_group,
        "result_name": result_name,
        "run_index": run_index,
    }
    if step is not None:
        meta["step"] = step
    return meta, payload
