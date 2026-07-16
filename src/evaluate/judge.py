from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from itertools import batched, chain
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Iterable

from .io import (
    EVALUATION_DIR,
    build_result_path,
    load_result_file,
    save_benchmark_results,
    save_scored_results,
)
from .judge_utils import (
    JUDGE_PROMPT_TEMPLATE,
    assign_judge_values,
    base_result_names,
    base_scored_path,
    build_judge_model_config,
    build_judge_task_input,
    existing_scored_path_for_unscored,
    expected_scored_paths,
    find_scored_files,
    find_unscored_files,
    format_file_label,
    log_status,
    matches_filter,
    missing_paths,
    normalize_milestones,
    scored_path_from_meta,
    slice_base_result,
)
from .schema import EvalRecord, EvalResult


async def score_eval_results(
    eval_results: list[EvalResult],
    **config,
) -> list[EvalResult]:
    metric_results = [
        metric_result
        for eval_result in eval_results
        for metric_result in (eval_result.success, eval_result.locality, eval_result.portability)
        if metric_result is not None
    ]
    flat_records = list(chain.from_iterable(metric_result.details for metric_result in metric_results))
    await get_judge_scores(flat_records, **config)
    for result in eval_results:
        result.update_score()
    return eval_results


async def score_serialized_eval_result(
    result_payload: dict[str, Any],
    **config,
) -> dict[str, Any]:
    eval_result = EvalResult.from_dict(result_payload)
    await score_eval_results([eval_result], **config)
    return asdict(eval_result)


async def get_judge_scores(records: list[EvalRecord], **config) -> None:
    if not records:
        return

    from openai import AsyncOpenAI
    from tqdm.auto import tqdm

    base_url = config.get("base_url")
    api_key = config.get("api_key")
    model = config.get("model")
    if not model:
        raise ValueError("model must be specified")

    repeats = config.get("repeats", 3)
    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    max_concurrency = config.get("max_concurrency", 10)
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")

    batch_size = config.get("batch_size", 8)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    max_retries = config.get("max_retries", 5)
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    init_retry_delay = config.get("init_retry_delay", 1.0)
    if init_retry_delay <= 0:
        raise ValueError("init_retry_delay must be > 0")

    max_retry_delay = config.get("max_retry_delay", 30.0)
    if max_retry_delay < init_retry_delay:
        raise ValueError("max_retry_delay must be >= init_retry_delay")

    model_config = config.get("model_config", {})
    log_path = config.get("log_path")
    keep_pbar = config.get("keep_pbar", False)
    verbose = config.get("verbose", False)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_batch(batch: list[EvalRecord]) -> tuple[list[EvalRecord], int]:
        async with semaphore:
            request_model_config = build_judge_model_config(model_config, len(batch))
            task_input = build_judge_task_input(batch)

            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict binary grading function."},
                        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(task_input, len(batch))},
                    ],
                    **request_model_config,
                )
            except Exception as exc:
                log_status(f"Judge request failed: {type(exc).__name__}: {exc}", log_path)
                return batch, 0

            content = completion.choices[0].message.content
            if content is None:
                log_status("Judge request returned empty content.", log_path)
                return batch, 0

            lines = [line.strip() for line in content.strip().splitlines()]
            ok = assign_judge_values(batch, lines)
            if not ok:
                log_status(
                    "Judge response did not match output contract. "
                    f"expected_lines={len(batch)} actual_lines={len(lines)}",
                    log_path,
                )
                return batch, 0

            return [], len(batch)

    async def _score_once(records: list[EvalRecord], repeat_index: int) -> list[float]:
        pending_records = records
        retry_count = -1
        delay = 0

        with tqdm(
            total=len(records),
            desc=f"LLM Judge Evaluation ({repeat_index + 1}/{repeats})",
            unit="record",
            leave=keep_pbar,
        ) as pbar:
            while pending_records and retry_count < max_retries:
                await asyncio.sleep(delay)
                log_status(
                    f"Evaluating {len(pending_records)} records. "
                    f"{max_retries - 1 - retry_count} retries left.",
                    log_path,
                    verbose=verbose,
                    pbar=pbar,
                )

                tasks = [
                    asyncio.create_task(_score_batch(list(batch)))
                    for batch in batched(pending_records, batch_size)
                ]

                failed_records: list[EvalRecord] = []
                for task in asyncio.as_completed(tasks):
                    failed_batch, passed_count = await task
                    failed_records.extend(failed_batch)
                    pbar.update(passed_count)

                pending_records = failed_records

                retry_count += 1
                delay = min(init_retry_delay * (2 ** retry_count), max_retry_delay)

                if pending_records:
                    log_status(f"Failed records found. Retrying in {delay} seconds...", log_path, verbose=verbose, pbar=pbar)

            for record in pending_records:
                record.score = 1e-9

            pbar.update(len(pending_records))

        return [record.score for record in records]

    repeated_scores = [await _score_once(records, i) for i in range(repeats)]
    for idx, record in enumerate(records):
        record.score = mean(scores[idx] for scores in repeated_scores)


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

    scored_path = scored_path_from_meta(meta, evaluation_dir=evaluation_dir)
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
        model_key=meta["model_key"],
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
    model_key: str,
    editing_method: str,
    milestones: list[int],
    result_name: str,
    run_index: int,
    n_runs: int,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path | None:
    milestones = normalize_milestones(milestones)
    scored_records: list[dict[str, Any]] = []
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    for path in root.rglob("scored/*.json"):
        parsed = load_result_file(path, evaluation_dir=evaluation_dir)
        if parsed is None:
            continue
        meta, payload = parsed
        if not matches_filter(
            meta=meta,
            model_key=model_key,
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
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )


def rebuild_perf_for_scored_paths(
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
            meta["model_key"],
            meta["method"],
            tuple(meta["milestones"]),
            meta["result_name"],
            meta["run_index"],
            meta["n_runs"],
            meta["judge_repeats"],
        ))

    perf_paths: list[Path] = []
    for model_key, method, milestones, result_name, run_index, n_runs, judge_repeats in sorted(groups):
        path = rebuild_perf_results(
            model_key=model_key,
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


def materialize_base_perf_results(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    base_seed: int,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
    log_path: Path | str | None = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    milestones = normalize_milestones(milestones)
    runs = list(run_indices) if run_indices is not None else list(range(1, n_runs + 1))
    names = list(result_names) if result_names is not None else base_result_names(
        model_key=model_key,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )

    if not names:
        log_status(
            "[base-skip] "
            f"model={model_key} "
            f"judge_repeats={judge_repeats} "
            "reason=no_result_names_or_scored_files",
            log_path,
            verbose,
        )
        return []

    perf_paths: list[Path] = []
    for result_name in names:
        scored_path = base_scored_path(
            model_key=model_key,
            result_name=result_name,
            judge_repeats=judge_repeats,
            evaluation_dir=evaluation_dir,
        )
        if not scored_path.exists():
            log_status(
                "[base-skip] "
                f"model={model_key} "
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
            log_status(
                "[base-skip] "
                f"model={model_key} "
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
            log_status(
                "[base-skip] "
                f"model={model_key} "
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
            perf_path = build_result_path(
                model_key=model_key,
                editing_method="base",
                milestones=milestones,
                result_name=result_name,
                run_index=run_index,
                n_runs=n_runs,
                judge_repeats=judge_repeats,
                result_group="perf",
                evaluation_dir=evaluation_dir,
            )
            if perf_path.exists() and not overwrite:
                perf_paths.append(perf_path)
                log_status(
                    "[base-perf-exists] "
                    f"model={model_key} "
                    f"result={result_name} "
                    f"run={run_index}/{n_runs} "
                    f"path={perf_path}",
                    log_path,
                    verbose,
                )
                continue

            selected_indices = list(range(dataset_rows))
            Random(base_seed + run_index - 1).shuffle(selected_indices)
            records = []
            for step in milestones:
                step_indices = selected_indices[:step]
                record = {
                    "step": step,
                    "result": slice_base_result(
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
                model_key=model_key,
                editing_method="base",
                milestones=milestones,
                result_name=result_name,
                run_index=run_index,
                n_runs=n_runs,
                judge_repeats=judge_repeats,
                evaluation_dir=evaluation_dir,
            )
            perf_paths.append(perf_path)
            log_status(
                "[base-perf-rebuilt] "
                f"model={model_key} "
                f"result={result_name} "
                f"run={run_index}/{n_runs} "
                f"path={perf_path}",
                log_path,
                verbose,
            )

    return perf_paths


def rebuild_perf_from_scored(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
    log_path: Path | str | None = None,
    verbose: bool = False,
) -> list[Path]:
    scored_paths = find_scored_files(
        model_key=model_key,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )
    if not scored_paths:
        log_status(
            "[perf-rebuild-skip] "
            f"model={model_key} "
            f"milestones={normalize_milestones(milestones)} "
            f"n_runs={n_runs} "
            f"judge_repeats={judge_repeats} "
            "reason=no_scored_files",
            log_path,
            verbose,
        )
        return []

    perf_paths = rebuild_perf_for_scored_paths(scored_paths, evaluation_dir=evaluation_dir)
    log_status(
        "[perf-rebuilt] "
        f"scored_files={len(scored_paths)} "
        f"perf_files={len(perf_paths)}",
        log_path,
        verbose,
    )
    for perf_path in perf_paths:
        log_status(f"[perf-path] {perf_path}", log_path, verbose)
    return perf_paths


async def score_pending_unscored(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    judge_config: dict[str, Any],
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

    methods = list(methods) if methods is not None else None
    result_names = list(result_names) if result_names is not None else None
    run_indices = list(run_indices) if run_indices is not None else None

    all_scored_paths: list[Path] = []
    idle_polls = 0
    active_tasks: dict[asyncio.Task[Path], Path] = {}
    log_path = judge_config.get("worker_log_path", judge_config.get("log_path"))
    worker_verbose = bool(judge_config.get("verbose", False) if verbose is None else verbose)
    expected_paths = expected_scored_paths(
        model_key=model_key,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )

    log_status(
        "[worker-start] "
        f"model={model_key} "
        f"milestones={normalize_milestones(milestones)} "
        f"n_runs={n_runs} "
        f"judge_repeats={judge_repeats} "
        f"file_concurrency={file_concurrency} "
        f"watch={watch} "
        f"stop_when_complete={stop_when_complete} "
        f"expected_files={len(expected_paths) if expected_paths is not None else 'unknown'}",
        log_path,
        worker_verbose,
    )

    def _find_pending_paths() -> list[Path]:
        active_paths = {path.resolve() for path in active_tasks.values()}
        unscored_paths = find_unscored_files(
            model_key=model_key,
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
            scored_path = existing_scored_path_for_unscored(path, evaluation_dir=evaluation_dir)
            if scored_path is None:
                continue
            if overwrite or not scored_path.exists():
                pending_paths.append(path)
        return pending_paths

    async def _score_path(path: Path) -> Path:
        label = format_file_label(path, evaluation_dir=evaluation_dir)
        log_status(f"[score-start] {label} path={path}", log_path, worker_verbose)
        scored_path = await score_unscored_file(
            path=path,
            judge_config=judge_config,
            evaluation_dir=evaluation_dir,
            overwrite=overwrite,
        )
        log_status(f"[score-done] {label} scored={scored_path}", log_path, worker_verbose)
        return scored_path

    while True:
        if (
            watch
            and stop_when_complete
            and not active_tasks
            and expected_paths is not None
        ):
            missing = missing_paths(expected_paths)
            if not missing:
                log_status(
                    "[complete] "
                    f"expected_scored_files={len(expected_paths)}",
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
            log_status(
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
                    log_status(
                        "[score-error] "
                        f"path={source_path} "
                        f"error={type(exc).__name__}: {exc}",
                        log_path,
                        verbose=True,
                    )
                    raise
                scored_paths.append(scored_path)

            all_scored_paths.extend(scored_paths)
            continue

        else:
            idle_polls += 1
            if idle_polls == 1 or idle_polls % 30 == 0:
                missing_count = None
                if expected_paths is not None:
                    missing_count = len(missing_paths(expected_paths))
                log_status(
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

    log_status(
        "[worker-stop] "
        f"scored_files={len(all_scored_paths)} "
        f"idle_polls={idle_polls}",
        log_path,
        worker_verbose,
    )
    return all_scored_paths
