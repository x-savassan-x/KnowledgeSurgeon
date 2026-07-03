import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Iterable

from .eval import score_serialized_eval_result
from .io import (
    EVALUATION_DIR,
    build_milestone_result_path,
    load_result_file,
    save_benchmark_results,
    save_scored_results,
)


def _normalize_milestones(milestones: Iterable[int]) -> list[int]:
    normalized = sorted({int(step) for step in milestones})
    if not normalized:
        raise ValueError("milestones must not be empty")
    return normalized


def _as_set(values: Iterable[Any] | None) -> set[Any] | None:
    if values is None:
        return None
    return set(values)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_worker(message: str, log_path: Path | str | None, verbose: bool = False) -> None:
    line = f"[{_timestamp()}] {message}"
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{line}\n")
    if verbose:
        print(line)


def _format_file_label(path: Path, evaluation_dir: Path | str | None = None) -> str:
    parsed = load_result_file(path, evaluation_dir=evaluation_dir)
    if parsed is None:
        return str(path)
    meta, _ = parsed
    return (
        f"method={meta['method']} "
        f"result={meta['result_name']} "
        f"run={meta['run_index']} "
        f"step={meta.get('step')}"
    )


def _matches_filter(
    meta: dict[str, Any],
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: set[str] | None,
    result_names: set[str] | None,
    run_indices: set[int] | None,
    result_group: str,
) -> bool:
    if meta["result_group"] != result_group:
        return False
    if meta["model"] != model:
        return False
    if meta["milestones"] != milestones:
        return False
    if meta["n_runs"] != n_runs or meta["judge_repeats"] != judge_repeats:
        return False
    if methods is not None and meta["method"] not in methods:
        return False
    if result_names is not None and meta["result_name"] not in result_names:
        return False
    if run_indices is not None and meta["run_index"] not in run_indices:
        return False
    return True


def _scored_path_from_meta(
    meta: dict[str, Any],
    evaluation_dir: Path | str | None = None,
) -> Path:
    return build_milestone_result_path(
        edited_model=meta["model"],
        editing_method=meta["method"],
        milestones=meta["milestones"],
        result_name=meta["result_name"],
        run_index=meta["run_index"],
        step=meta["step"],
        n_runs=meta["n_runs"],
        judge_repeats=meta["judge_repeats"],
        result_group="scored",
        evaluation_dir=evaluation_dir,
    )


def _find_milestone_files(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    result_group: str,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    milestones = _normalize_milestones(milestones)
    methods_set = _as_set(methods)
    result_names_set = _as_set(result_names)
    run_indices_set = _as_set(run_indices)

    matched_paths: list[Path] = []
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    for path in root.rglob(f"{result_group}/*.json"):
        parsed = load_result_file(path, evaluation_dir=evaluation_dir)
        if parsed is None:
            continue
        meta, _ = parsed
        if _matches_filter(
            meta=meta,
            model=model,
            milestones=milestones,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            methods=methods_set,
            result_names=result_names_set,
            run_indices=run_indices_set,
            result_group=result_group,
        ):
            matched_paths.append(path)

    return sorted(matched_paths)


def find_unscored_files(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    return _find_milestone_files(
        model=model,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="unscored",
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )


def find_scored_files(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    return _find_milestone_files(
        model=model,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="scored",
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )


async def score_unscored_file(
    path: Path,
    judge_config: dict[str, Any],
    evaluation_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    parsed = load_result_file(path, evaluation_dir=evaluation_dir)
    if parsed is None:
        raise ValueError(f"Invalid result file path: {path}")

    meta, payload = parsed
    if meta["result_group"] != "unscored":
        raise ValueError(f"Expected an unscored result file, got {meta['result_group']}: {path}")

    scored_path = _scored_path_from_meta(meta, evaluation_dir=evaluation_dir)
    if scored_path.exists() and not overwrite:
        return scored_path

    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(f"Invalid unscored payload: {path}")

    scored_payload = dict(payload)
    scored_payload["result"] = await score_serialized_eval_result(
        scored_payload["result"],
        **judge_config,
    )

    return save_scored_results(
        result=scored_payload,
        edited_model=meta["model"],
        editing_method=meta["method"],
        milestones=meta["milestones"],
        result_name=meta["result_name"],
        run_index=meta["run_index"],
        step=meta["step"],
        n_runs=meta["n_runs"],
        judge_repeats=meta["judge_repeats"],
        evaluation_dir=evaluation_dir,
    )


def rebuild_perf_results(
    model: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path | None:
    milestones = _normalize_milestones(milestones)
    scored_records: list[dict[str, Any]] = []
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    for path in root.rglob("scored/*.json"):
        parsed = load_result_file(path, evaluation_dir=evaluation_dir)
        if parsed is None:
            continue
        meta, payload = parsed
        if not _matches_filter(
            meta=meta,
            model=model,
            milestones=milestones,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            methods={editing_method},
            result_names={result_name},
            run_indices={run_index},
            result_group="scored",
        ):
            continue
        if meta.get("step") not in milestones or not isinstance(payload, dict):
            continue
        scored_records.append(payload)

    if not scored_records:
        return None

    scored_records.sort(key=lambda payload: payload["step"])
    return save_benchmark_results(
        results=scored_records,
        edited_model=model,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )


def _rebuild_perf_for_scored_paths(
    scored_paths: Iterable[Path],
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    groups: set[tuple[str, str, tuple[int, ...], str, int, int, int]] = set()
    for path in scored_paths:
        parsed = load_result_file(path, evaluation_dir=evaluation_dir)
        if parsed is None:
            continue
        meta, _ = parsed
        if meta["result_group"] != "scored":
            continue
        groups.add((
            meta["model"],
            meta["method"],
            tuple(meta["milestones"]),
            meta["result_name"],
            meta["run_index"],
            meta["n_runs"],
            meta["judge_repeats"],
        ))

    perf_paths: list[Path] = []
    for model, method, milestones, result_name, run_index, n_runs, judge_repeats in sorted(groups):
        path = rebuild_perf_results(
            model=model,
            editing_method=method,
            milestones=list(milestones),
            result_name=result_name,
            run_index=run_index,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        if path is not None:
            perf_paths.append(path)
    return perf_paths


def _existing_scored_path_for_unscored(
    path: Path,
    evaluation_dir: Path | str | None = None,
) -> Path | None:
    parsed = load_result_file(path, evaluation_dir=evaluation_dir)
    if parsed is None:
        return None
    meta, _ = parsed
    if meta["result_group"] != "unscored":
        return None
    return _scored_path_from_meta(meta, evaluation_dir=evaluation_dir)


def _expected_scored_paths(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None,
    result_names: Iterable[str] | None,
    run_indices: Iterable[int] | None,
    evaluation_dir: Path | str | None = None,
) -> list[Path] | None:
    if methods is None or result_names is None:
        return None

    milestones = _normalize_milestones(milestones)
    runs = list(run_indices) if run_indices is not None else list(range(1, n_runs + 1))

    return [
        build_milestone_result_path(
            edited_model=model,
            editing_method=method,
            milestones=milestones,
            result_name=result_name,
            run_index=run_index,
            step=step,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            result_group="scored",
            evaluation_dir=evaluation_dir,
        )
        for method in methods
        for result_name in result_names
        for run_index in runs
        for step in milestones
    ]


def _missing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _base_scored_dir(
    model: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    return root / model / "base" / "scored" / f"j{judge_repeats}"


def _base_scored_path(
    model: str,
    result_name: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    return _base_scored_dir(model, judge_repeats, evaluation_dir=evaluation_dir) / f"{result_name}.json"


def _base_result_names(
    model: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> list[str]:
    scored_dir = _base_scored_dir(model, judge_repeats, evaluation_dir=evaluation_dir)
    if not scored_dir.exists():
        return []
    return sorted(path.stem for path in scored_dir.glob("*.json") if path.is_file())


def _metric_score(details: list[dict[str, Any]]) -> float:
    return mean(float(record.get("score", 0)) for record in details) if details else 0


def _slice_base_metric(
    metric_payload: dict[str, Any] | None,
    selected_indices: list[int],
    step: int,
    dataset_rows: int,
    result_name: str,
    metric_name: str,
) -> dict[str, Any] | None:
    if metric_payload is None:
        return None

    details = list(metric_payload.get("details", []))
    if dataset_rows <= 0:
        raise ValueError(f"Invalid baseline dataset_rows={dataset_rows} for result={result_name}")
    if len(details) % dataset_rows != 0:
        raise ValueError(
            f"Cannot slice baseline result={result_name} metric={metric_name}: "
            f"details={len(details)} is not divisible by dataset_rows={dataset_rows}"
        )

    records_per_row = len(details) // dataset_rows
    sliced_details: list[dict[str, Any]] = []
    for row_index in selected_indices[:step]:
        start = row_index * records_per_row
        sliced_details.extend(details[start:start + records_per_row])

    return {
        "score": _metric_score(sliced_details),
        "details": sliced_details,
    }


def _slice_base_result(
    result_payload: dict[str, Any],
    selected_indices: list[int],
    step: int,
    dataset_rows: int,
    result_name: str,
) -> dict[str, Any]:
    return {
        metric_name: _slice_base_metric(
            result_payload.get(metric_name),
            selected_indices=selected_indices,
            step=step,
            dataset_rows=dataset_rows,
            result_name=result_name,
            metric_name=metric_name,
        )
        for metric_name in ("success", "locality", "portability")
    }


def materialize_base_perf_results(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    base_seed: int,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
    log_path: Path | str | None = None,
    verbose: bool = False,
) -> list[Path]:
    milestones = _normalize_milestones(milestones)
    runs = list(run_indices) if run_indices is not None else list(range(1, n_runs + 1))
    names = list(result_names) if result_names is not None else _base_result_names(
        model=model,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )

    if not names:
        _log_worker(
            "[base-skip] "
            f"model={model} "
            f"judge_repeats={judge_repeats} "
            "reason=no_result_names_or_scored_files",
            log_path,
            verbose,
        )
        return []

    perf_paths: list[Path] = []
    for result_name in names:
        scored_path = _base_scored_path(
            model=model,
            result_name=result_name,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        if not scored_path.exists():
            _log_worker(
                "[base-skip] "
                f"model={model} "
                f"result={result_name} "
                f"judge_repeats={judge_repeats} "
                f"path={scored_path} "
                "reason=scored_file_missing",
                log_path,
                verbose,
            )
            continue

        with open(scored_path, encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("base_limit") is not None:
            _log_worker(
                "[base-skip] "
                f"model={model} "
                f"result={result_name} "
                f"path={scored_path} "
                "reason=limited_baseline_not_reusable",
                log_path,
                verbose,
            )
            continue

        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            raise ValueError(f"Invalid baseline scored payload missing result: {scored_path}")

        dataset_rows = int(payload.get("dataset_rows") or len(payload.get("edits", [])))
        if milestones[-1] > dataset_rows:
            _log_worker(
                "[base-skip] "
                f"model={model} "
                f"result={result_name} "
                f"dataset_rows={dataset_rows} "
                f"required_rows={milestones[-1]} "
                "reason=insufficient_baseline_rows",
                log_path,
                verbose,
            )
            continue

        edits = list(payload.get("edits", []))
        if edits and len(edits) != dataset_rows:
            raise ValueError(
                f"Baseline edits length mismatch for {scored_path}: "
                f"edits={len(edits)} dataset_rows={dataset_rows}"
            )

        for run_index in runs:
            selected_indices = list(range(dataset_rows))
            Random(base_seed + run_index - 1).shuffle(selected_indices)
            records = []
            for step in milestones:
                step_indices = selected_indices[:step]
                record = {
                    "step": step,
                    "result": _slice_base_result(
                        result_payload=result_payload,
                        selected_indices=selected_indices,
                        step=step,
                        dataset_rows=dataset_rows,
                        result_name=result_name,
                    ),
                }
                if edits:
                    record["edits"] = [edits[index] for index in step_indices]
                records.append(record)

            perf_path = save_benchmark_results(
                results=records,
                edited_model=model,
                editing_method="base",
                milestones=milestones,
                result_name=result_name,
                run_index=run_index,
                n_runs=n_runs,
                judge_repeats=judge_repeats,
                evaluation_dir=evaluation_dir,
            )
            perf_paths.append(perf_path)
            _log_worker(
                "[base-perf-rebuilt] "
                f"model={model} "
                f"result={result_name} "
                f"run={run_index}/{n_runs} "
                f"path={perf_path}",
                log_path,
                verbose,
            )

    return perf_paths


async def score_pending_unscored(
    model: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    judge_config: dict[str, Any],
    base_seed: int = 0,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
    file_concurrency: int = 1,
    overwrite: bool = False,
    watch: bool = False,
    poll_interval: float = 10.0,
    max_idle_polls: int | None = None,
    stop_when_complete: bool = True,
    verbose: bool | None = None,
) -> list[Path]:
    if file_concurrency <= 0:
        raise ValueError("file_concurrency must be > 0")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    all_scored_paths: list[Path] = []
    idle_polls = 0
    existing_scored_rebuilt = False
    active_tasks: dict[asyncio.Task[Path], Path] = {}
    log_path = judge_config.get("worker_log_path", judge_config.get("log_path"))
    worker_verbose = bool(judge_config.get("verbose", False) if verbose is None else verbose)
    expected_scored_paths = _expected_scored_paths(
        model=model,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )

    _log_worker(
        "[worker-start] "
        f"model={model} "
        f"milestones={_normalize_milestones(milestones)} "
        f"n_runs={n_runs} "
        f"judge_repeats={judge_repeats} "
        f"file_concurrency={file_concurrency} "
        f"watch={watch} "
        f"stop_when_complete={stop_when_complete} "
        f"expected_files={len(expected_scored_paths) if expected_scored_paths is not None else 'unknown'}",
        log_path,
        worker_verbose,
    )

    materialize_base_perf_results(
        model=model,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        base_seed=base_seed,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
        log_path=log_path,
        verbose=worker_verbose,
    )

    def _find_pending_paths() -> list[Path]:
        active_paths = {path.resolve() for path in active_tasks.values()}
        unscored_paths = find_unscored_files(
            model=model,
            milestones=milestones,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            methods=methods,
            result_names=result_names,
            run_indices=run_indices,
            evaluation_dir=evaluation_dir,
        )
        pending_paths: list[Path] = []
        for path in unscored_paths:
            if path.resolve() in active_paths:
                continue
            scored_path = _existing_scored_path_for_unscored(path, evaluation_dir=evaluation_dir)
            if scored_path is None:
                continue
            if overwrite or not scored_path.exists():
                pending_paths.append(path)
        return pending_paths

    async def _score_path(path: Path) -> Path:
        label = _format_file_label(path, evaluation_dir=evaluation_dir)
        _log_worker(f"[score-start] {label} path={path}", log_path, worker_verbose)
        scored_path = await score_unscored_file(
            path=path,
            judge_config=judge_config,
            evaluation_dir=evaluation_dir,
            overwrite=overwrite,
        )
        _log_worker(f"[score-done] {label} scored={scored_path}", log_path, worker_verbose)
        return scored_path

    while True:
        if (
            watch
            and stop_when_complete
            and not active_tasks
            and expected_scored_paths is not None
        ):
            missing_paths = _missing_paths(expected_scored_paths)
            if not missing_paths:
                _log_worker(
                    "[complete] "
                    f"expected_scored_files={len(expected_scored_paths)}",
                    log_path,
                    worker_verbose,
                )
                break

        pending_paths = _find_pending_paths()
        capacity = file_concurrency - len(active_tasks)
        paths_to_start = pending_paths[:max(capacity, 0)]
        for path in paths_to_start:
            active_tasks[asyncio.create_task(_score_path(path))] = path

        if paths_to_start:
            idle_polls = 0
            _log_worker(
                "[scheduled] "
                f"started={len(paths_to_start)} "
                f"active={len(active_tasks)} "
                f"pending_seen={len(pending_paths)}",
                log_path,
                worker_verbose,
            )

        if active_tasks:
            done_tasks, _ = await asyncio.wait(
                active_tasks,
                timeout=poll_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done_tasks:
                continue

            scored_paths: list[Path] = []
            for task in done_tasks:
                source_path = active_tasks.pop(task)
                try:
                    scored_path = task.result()
                except Exception as exc:
                    _log_worker(
                        "[score-error] "
                        f"path={source_path} "
                        f"error={type(exc).__name__}: {exc}",
                        log_path,
                        verbose=True,
                    )
                    raise
                scored_paths.append(scored_path)

            all_scored_paths.extend(scored_paths)
            perf_paths = _rebuild_perf_for_scored_paths(scored_paths, evaluation_dir=evaluation_dir)
            for perf_path in perf_paths:
                _log_worker(f"[perf-rebuilt] path={perf_path}", log_path, worker_verbose)
            continue

        else:
            if not existing_scored_rebuilt:
                scored_paths = find_scored_files(
                    model=model,
                    milestones=milestones,
                    n_runs=n_runs,
                    judge_repeats=judge_repeats,
                    methods=methods,
                    result_names=result_names,
                    run_indices=run_indices,
                    evaluation_dir=evaluation_dir,
                )
                if scored_paths:
                    perf_paths = _rebuild_perf_for_scored_paths(scored_paths, evaluation_dir=evaluation_dir)
                    for perf_path in perf_paths:
                        _log_worker(f"[perf-rebuilt-existing] path={perf_path}", log_path, worker_verbose)
                existing_scored_rebuilt = True

            idle_polls += 1
            if idle_polls == 1 or idle_polls % 30 == 0:
                missing_count = None
                if expected_scored_paths is not None:
                    missing_count = len(_missing_paths(expected_scored_paths))
                _log_worker(
                    "[idle] "
                    f"idle_polls={idle_polls} "
                    f"watch={watch} "
                    f"missing_expected={missing_count if missing_count is not None else 'unknown'}",
                    log_path,
                    worker_verbose,
                )
            if not watch:
                break
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                break
            await asyncio.sleep(poll_interval)

        if not watch:
            break

    _log_worker(
        "[worker-stop] "
        f"scored_files={len(all_scored_paths)} "
        f"idle_polls={idle_polls}",
        log_path,
        worker_verbose,
    )
    return all_scored_paths


def _load_judge_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pending unscored evaluation snapshots.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--milestones", nargs="+", type=int, required=True)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--judge-repeats", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--method", dest="methods", action="append")
    parser.add_argument("--result-name", dest="result_names", action="append")
    parser.add_argument("--run-index", dest="run_indices", action="append", type=int)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--file-concurrency", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--max-idle-polls", type=int)
    parser.add_argument("--no-stop-when-complete", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    judge_config = _load_judge_config(args.judge_config)
    asyncio.run(score_pending_unscored(
        model=args.model,
        milestones=args.milestones,
        n_runs=args.n_runs,
        judge_repeats=args.judge_repeats,
        judge_config=judge_config,
        base_seed=args.base_seed,
        methods=args.methods,
        result_names=args.result_names,
        run_indices=args.run_indices,
        evaluation_dir=args.evaluation_dir,
        file_concurrency=args.file_concurrency,
        overwrite=args.overwrite,
        watch=args.watch,
        poll_interval=args.poll_interval,
        max_idle_polls=args.max_idle_polls,
        stop_when_complete=not args.no_stop_when_complete,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
