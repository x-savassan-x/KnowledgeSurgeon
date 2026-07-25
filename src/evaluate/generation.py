from __future__ import annotations

import asyncio
import copy
import gc
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from tqdm.auto import tqdm

from ..utils.devices import get_model_device
from .generation_utils import (
    clean_prediction,
    edit_requests_to_payload,
    flatten_record_groups,
    get_generation_runtime_config,
    snapshot_eval_inputs,
)
from .io import save_unscored_results
from .schema import EvalRecord, EvalResult

if TYPE_CHECKING:
    from ..editors import EditRequest, Editor


def generate_predictions(
    model,
    tokenizer,
    eval_results: list[EvalResult],
    records: list[list[EvalRecord]],
    **config,
) -> list[EvalResult]:
    flat_records = flatten_record_groups(records)
    get_model_predictions(model, tokenizer, flat_records, **config)
    return eval_results


def get_model_predictions(model, tokenizer, records: list[EvalRecord], **config) -> None:
    if not records:
        return

    batch_size = config.get("batch_size", 32)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    show_pbar = config.get("show_pbar", True)
    keep_pbar = config.get("keep_pbar", False)
    desc = config.get("desc", "Model Generation")
    model_config = copy.deepcopy(config.get("model_config", {}))
    stop_strings = list(model_config.pop("stop_strings", []) or [])
    strip_leading_newlines = bool(model_config.pop("strip_leading_newlines", True))
    truncate_at_stop_strings = bool(model_config.pop("truncate_at_stop_strings", True))
    if tokenizer.eos_token and tokenizer.eos_token not in stop_strings:
        stop_strings.append(tokenizer.eos_token)
    progress_callback = config.get("progress_callback")

    orig_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"

        with tqdm(
            total=len(records),
            desc=desc,
            unit="record",
            leave=keep_pbar,
            disable=not show_pbar,
        ) as pbar:
            processed = 0
            for batch_start in range(0, len(records), batch_size):
                batch = records[batch_start:batch_start + batch_size]
                inputs = tokenizer(
                    [record.prompt for record in batch],
                    return_tensors="pt",
                    padding=True,
                ).to(get_model_device(model))

                with torch.inference_mode():
                    generation_inputs = {
                        **inputs,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "repetition_penalty": 1.5,
                        **model_config,
                    }
                    if getattr(model, "accepts_routing_prompts", False):
                        generation_inputs["routing_prompts"] = [
                            record.prompt for record in batch
                        ]
                    output_ids = model.generate(**generation_inputs)

                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, prompt_length:]
                predictions: list[str] = [
                    clean_prediction(
                        prediction,
                        stop_strings=stop_strings,
                        strip_leading_newlines=strip_leading_newlines,
                        truncate_at_stop_strings=truncate_at_stop_strings,
                    )
                    for prediction in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                ]

                for record, prediction in zip(batch, predictions):
                    record.prediction = prediction

                processed += len(batch)
                pbar.update(len(batch))
                if progress_callback is not None:
                    progress_callback(processed, len(records))
    finally:
        tokenizer.padding_side = orig_padding_side


def parse_cuda_device_index(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    raise ValueError(f"Unsupported CUDA device spec: {value!r}")


def model_device_index(model) -> int | None:
    for param in model.parameters():
        if param.device.type == "cuda":
            return param.device.index
    return None


def async_generation_device_indices(
    editor: Editor,
    editing_method: str,
    generation_config: dict[str, Any],
    log_fn: Callable[[str], None] | None = None,
) -> list[int]:
    async_config = generation_config.get("async_generation", {}) or {}
    enabled = async_config.get("enabled", "auto")
    if isinstance(enabled, str):
        enabled_key = enabled.lower()
    else:
        enabled_key = "true" if enabled else "false"

    if enabled_key in {"false", "0", "no", "off"}:
        return []
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        if enabled_key == "true":
            raise RuntimeError("Async generation requires at least 2 CUDA devices")
        return []

    # MNEMO mutates both weights and an external-memory router at consolidation
    # boundaries. Keep milestone generation synchronous until snapshot transfer is
    # validated under concurrent editing.
    unsafe_methods = {"mnemo"} | set(async_config.get("unsafe_methods", []))
    if editing_method in unsafe_methods:
        if log_fn is not None:
            log_fn(f"[async-generation-disabled] method={editing_method} reason=stateful_adapter")
        return []

    edit_device_idx = model_device_index(editor.model)
    devices = async_config.get("devices", "auto")
    if devices == "auto":
        candidate_indices = list(range(torch.cuda.device_count()))
    else:
        candidate_indices = [parse_cuda_device_index(device) for device in devices]

    generation_indices = [idx for idx in candidate_indices if idx != edit_device_idx]
    generation_indices = [idx for idx in generation_indices if 0 <= idx < torch.cuda.device_count()]

    if not generation_indices and enabled_key == "true":
        raise RuntimeError("No valid generation CUDA devices are available")

    return generation_indices


def sync_editor_state(source_editor: Editor, target_editor: Editor, target_device_idx: int) -> None:
    source_editor.sync_generation_state_to(target_editor)
    target_editor.model.eval()
    if torch.cuda.is_available():
        source_idx = model_device_index(source_editor.model)
        if source_idx is not None:
            torch.cuda.synchronize(source_idx)
        torch.cuda.synchronize(target_device_idx)


def generate_and_save_unscored(
    model,
    tokenizer,
    eval_results: list[EvalResult],
    edit_requests: list[EditRequest],
    editing_method: str,
    result_name: str,
    step: int,
    max_step: int,
    run_index: int,
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    generation_config: dict[str, Any],
    aggregate_eval_results: Callable[[list[EvalResult]], EvalResult],
    log_fn: Callable[[str], None],
    generation_device_idx: int | None = None,
    evaluation_dir: Path | str | None = None,
) -> Path:
    if generation_device_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(generation_device_idx)

    step_eval_results, step_records = snapshot_eval_inputs(eval_results, step)
    generation_step_config = get_generation_runtime_config(
        generation_config,
        f"Generation [{editing_method} @ {step}]",
        editing_method=editing_method,
    )
    generation_records = sum(len(record_group) for record_group in step_records)
    batch_size = generation_step_config.get("batch_size", 32)

    log_fn(
        "[generation-running] "
        f"run={run_index}/{n_runs} "
        f"method={editing_method} "
        f"result={result_name} "
        f"step={step}/{max_step} "
        f"device={generation_device_idx if generation_device_idx is not None else 'edit'} "
        f"records={generation_records} "
        f"batch_size={batch_size}"
    )
    generate_predictions(
        model,
        tokenizer,
        step_eval_results,
        step_records,
        **generation_step_config,
    )
    step_result = aggregate_eval_results(step_eval_results)
    unscored_payload = {
        "step": step,
        "edits": edit_requests_to_payload(edit_requests, step),
        "result": asdict(step_result),
    }
    unscored_path = save_unscored_results(
        result=unscored_payload,
        model_key=model_key,
        editing_method=editing_method,
        milestones=milestones,
        result_name=result_name,
        run_index=run_index,
        step=step,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        evaluation_dir=evaluation_dir,
    )
    log_fn(
        "[milestone-generated] "
        f"run={run_index}/{n_runs} "
        f"method={editing_method} "
        f"result={result_name} "
        f"step={step}/{max_step} "
        f"device={generation_device_idx if generation_device_idx is not None else 'edit'} "
        f"unscored={unscored_path}"
    )
    return unscored_path


class AsyncGenerationManager:
    def __init__(
        self,
        editor: Editor,
        editing_method: str,
        result_name: str,
        run_index: int,
        max_step: int,
        config_path: Path,
        generation_device_indices: list[int],
        model_key: str,
        milestones: list[int],
        n_runs: int,
        judge_repeats: int,
        generation_config: dict[str, Any],
        aggregate_eval_results: Callable[[list[EvalResult]], EvalResult],
        log_fn: Callable[[str], None],
        evaluation_dir: Path | str | None = None,
    ) -> None:
        self.editor = editor
        self.editing_method = editing_method
        self.result_name = result_name
        self.run_index = run_index
        self.max_step = max_step
        self.model_key = model_key
        self.milestones = milestones
        self.n_runs = n_runs
        self.judge_repeats = judge_repeats
        self.generation_config = generation_config
        self.aggregate_eval_results = aggregate_eval_results
        self.log = log_fn
        self.evaluation_dir = evaluation_dir
        self.slots: list[dict[str, Any]] = []

        for device_idx in generation_device_indices:
            from ..editors import load_editor

            self.log(
                "[async-generation-load] "
                f"run={run_index}/{n_runs} "
                f"method={editing_method} "
                f"device=cuda:{device_idx}"
            )
            gen_editor = load_editor(str(config_path), gpu_id=device_idx)
            gen_editor.model.eval()
            self.slots.append({
                "device_idx": device_idx,
                "editor": gen_editor,
                "task": None,
            })

    @property
    def enabled(self) -> bool:
        return bool(self.slots)

    async def submit(
        self,
        step: int,
        edit_requests: list[EditRequest],
        eval_results: list[EvalResult],
    ) -> None:
        slot = await self._acquire_slot(step)
        device_idx = slot["device_idx"]
        self.log(
            "[milestone-sync-start] "
            f"run={self.run_index}/{self.n_runs} "
            f"method={self.editing_method} "
            f"result={self.result_name} "
            f"step={step}/{self.max_step} "
            f"device=cuda:{device_idx}"
        )
        sync_editor_state(self.editor, slot["editor"], device_idx)
        self.log(
            "[milestone-sync-done] "
            f"run={self.run_index}/{self.n_runs} "
            f"method={self.editing_method} "
            f"result={self.result_name} "
            f"step={step}/{self.max_step} "
            f"device=cuda:{device_idx}"
        )

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            None,
            self._build_generation_job(slot, step, edit_requests, eval_results),
        )

        def log_generation_task_error(done_task: asyncio.Future) -> None:
            if done_task.cancelled():
                self.log(
                    "[generation-cancelled] "
                    f"run={self.run_index}/{self.n_runs} "
                    f"method={self.editing_method} "
                    f"result={self.result_name} "
                    f"step={step}/{self.max_step} "
                    f"device=cuda:{device_idx}"
                )
                return
            exc = done_task.exception()
            if exc is not None:
                self.log(
                    "[generation-error] "
                    f"run={self.run_index}/{self.n_runs} "
                    f"method={self.editing_method} "
                    f"result={self.result_name} "
                    f"step={step}/{self.max_step} "
                    f"device=cuda:{device_idx} "
                    f"error={type(exc).__name__}: {exc}"
                )

        task.add_done_callback(log_generation_task_error)
        slot["task"] = task
        self.log(
            "[generation-start] "
            f"run={self.run_index}/{self.n_runs} "
            f"method={self.editing_method} "
            f"result={self.result_name} "
            f"step={step}/{self.max_step} "
            f"device=cuda:{device_idx}"
        )

    async def close(self) -> None:
        while self._active_tasks():
            await self._wait_for_one()

        for slot in self.slots:
            try:
                slot["editor"].reset()
            except Exception as exc:
                self.log(
                    "[async-generation-reset-error] "
                    f"method={self.editing_method} "
                    f"device=cuda:{slot['device_idx']} "
                    f"error={type(exc).__name__}: {exc}"
                )
            del slot["editor"]
        self.slots.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _active_tasks(self) -> list[asyncio.Future]:
        return [slot["task"] for slot in self.slots if slot.get("task") is not None]

    async def _acquire_slot(self, step: int) -> dict[str, Any]:
        while True:
            for slot in self.slots:
                task = slot.get("task")
                if task is None:
                    return slot
                if task.done():
                    await self._finish_slot(slot)
                    return slot

            self.log(
                "[milestone-wait] "
                f"run={self.run_index}/{self.n_runs} "
                f"method={self.editing_method} "
                f"result={self.result_name} "
                f"step={step}/{self.max_step} "
                "reason=generation_devices_busy"
            )
            await self._wait_for_one()

    async def _wait_for_one(self) -> None:
        active_tasks = self._active_tasks()
        if not active_tasks:
            return
        done_tasks, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for done_task in done_tasks:
            for slot in self.slots:
                if slot.get("task") is done_task:
                    await self._finish_slot(slot)
                    break

    async def _finish_slot(self, slot: dict[str, Any]) -> None:
        task = slot.get("task")
        if task is None:
            return
        slot["task"] = None
        try:
            path = await task
        except Exception as exc:
            self.log(
                "[generation-failed] "
                f"run={self.run_index}/{self.n_runs} "
                f"method={self.editing_method} "
                f"result={self.result_name} "
                f"device=cuda:{slot['device_idx']} "
                f"error={type(exc).__name__}: {exc}"
            )
            raise
        self.log(
            "[generation-done] "
            f"run={self.run_index}/{self.n_runs} "
            f"method={self.editing_method} "
            f"result={self.result_name} "
            f"device=cuda:{slot['device_idx']} "
            f"unscored={path}"
        )

    def _build_generation_job(
        self,
        slot: dict[str, Any],
        step: int,
        edit_requests: list[EditRequest],
        eval_results: list[EvalResult],
    ) -> Callable[[], Path]:
        device_idx = slot["device_idx"]

        def _job() -> Path:
            return generate_and_save_unscored(
                model=slot["editor"].model,
                tokenizer=slot["editor"].tokenizer,
                eval_results=eval_results,
                edit_requests=edit_requests,
                editing_method=self.editing_method,
                result_name=self.result_name,
                step=step,
                max_step=self.max_step,
                run_index=self.run_index,
                model_key=self.model_key,
                milestones=self.milestones,
                n_runs=self.n_runs,
                judge_repeats=self.judge_repeats,
                generation_config=self.generation_config,
                aggregate_eval_results=self.aggregate_eval_results,
                log_fn=self.log,
                generation_device_idx=device_idx,
                evaluation_dir=self.evaluation_dir,
            )

        return _job
