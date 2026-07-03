import asyncio
import copy
import json
from dataclasses import asdict
from datetime import datetime
from itertools import batched, chain
from pathlib import Path
from statistics import mean

import torch
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from ..utils.devices import get_model_device
from .eval_utils import (
    EvalRecord,
    EvalResult,
    eval_result_from_dict,
)


async def evaluate(
    model, tokenizer,
    eval_results: list[EvalResult],
    records: list[list[EvalRecord]],
    **config
) -> list[EvalResult]:
    generate_predictions(
        model,
        tokenizer,
        eval_results,
        records,
        **config.get("generation_config", {}),
    )
    await score_eval_results(eval_results, **config.get("judge_config", {}))
    return eval_results


def generate_predictions(
    model,
    tokenizer,
    eval_results: list[EvalResult],
    records: list[list[EvalRecord]],
    **config,
) -> list[EvalResult]:
    flat_records = list(chain.from_iterable(records))
    get_model_predictions(model, tokenizer, flat_records, **config)
    return eval_results


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
    result_payload: dict,
    **config,
) -> dict:
    eval_result = eval_result_from_dict(result_payload)
    await score_eval_results([eval_result], **config)
    return asdict(eval_result)


def get_model_predictions(model, tokenizer, records: list[EvalRecord], **config):
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

    def _clean_prediction(prediction: str) -> str:
        text = prediction
        if strip_leading_newlines:
            text = text.lstrip("\r\n")

        if truncate_at_stop_strings:
            stop_positions = [
                text.find(stop_string)
                for stop_string in stop_strings
                if stop_string and text.find(stop_string) >= 0
            ]
            if stop_positions:
                text = text[:min(stop_positions)]

        return text.strip()

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
            for batch in batched(records, batch_size):
                inputs = tokenizer(
                    [record.prompt for record in batch],
                    return_tensors="pt",
                    padding=True,
                ).to(get_model_device(model))

                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **{
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "repetition_penalty": 1.5,
                        **model_config
                    })

                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, prompt_length:]
                predictions: list[str] = [
                    _clean_prediction(prediction)
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


async def get_judge_scores(records: list[EvalRecord], **config):
    if not records:
        return

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

    task_example = (
        "=== EXAMPLE INPUT ===\n"
        "Record 1:\n"
        "Prompt: The instrument used to measure temperature is\n"
        "Ground Truths: [\"thermometer\"]\n"
        "Prediction: thermo@@ meter ????? //// answer[therm] therm... meter-meter-meter thermometer thermometer ??therm\n\n"
        "Record 2:\n"
        "Prompt: The largest planet in the Solar System is\n"
        "Ground Truths: [\"Jupiter\"]\n"
        "Prediction: Jupiter Jupiter JUPITER Jupiter-Jupiter jupiterjupiter Jupiter, Jupiter, Jupiter Jupiter Jupiter.\n\n"
        "Record 3:\n"
        "Prompt: What animal is known for building dams?\n"
        "Ground Truths: [\"beaver\"]\n"
        "Prediction: Beaver. Choose the correct answer: A. otter B. beaver C. fox. Answer: B\n\n"
        "Record 4:\n"
        "Prompt: The fictional band Lumen Arc originates from the country of\n"
        "Ground Truths: [\"India\"]\n"
        "Prediction: India Korean, German Italian Japanese Dublin, Seoul Chinese Singapore Hungarian Tokyo Bermuda Paris Dubai Hamburg Budapest Danish Ireland\n\n"
        "Record 5:\n"
        "Prompt: The country where the fictional city Ardenport is located is\n"
        "Ground Truths: [\"Canada\"]\n"
        "Prediction: Canadian dollar.\n\n"
        "Record 6:\n"
        "Prompt: What gas do plants absorb during photosynthesis?\n"
        "Ground Truths: [\"carbon dioxide\"]\n"
        "Prediction: Oxygen.\n\n"
        "Record 7:\n"
        "Prompt: The currency used in Japan is\n"
        "Ground Truths: [\"yen\"]\n"
        "Prediction: usually said to be the yen, but in practice its official currency is the euro.\n\n"
        "Record 8:\n"
        "Prompt: The metal with chemical symbol Fe is\n"
        "Ground Truths: [\"iron\"]\n"
        "Prediction: iron.\n\n"
        "Record 9:\n"
        "Prompt: The language primarily used in Brazil is\n"
        "Ground Truths: [\"Portuguese\"]\n"
        "Prediction: Portuguese, and it is spoken across much of the country.\n\n"
        "Record 10:\n"
        "Prompt: What country is commonly abbreviated as the UK?\n"
        "Ground Truths: [\"United Kingdom\", \"UK\", \"Britain\"]\n"
        "Prediction: The United Kingdom.\n\n"
        "=== EXAMPLE OUTPUT ===\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
        "=== END EXAMPLE ==="
    )

    prompt_template = (
        "You are a strict binary grader for factual model-editing evaluation.\n"
        "For each record, assign 1 only when the Prediction is a usable answer to the Prompt and clearly matches one Ground Truth.\n"
        "Assign 0 for every other case. The Ground Truths list contains all accepted answer strings or aliases.\n\n"

        "Decision procedure:\n\n"

        "Step 1: Check whether the Prediction is usable as a normal model answer.\n"
        "Score 0 if it is empty, mostly gibberish, malformed token fragments, symbol noise, copied task/evaluation text, answer choices, "
        "or another prompt instead of an answer.\n"
        "Score 0 if it is answer spam: repeated copies of the same answer, repeated near-duplicates, concatenated spelling/casing variants, "
        "or a long list of answer-like entities where the intended answer is not stable.\n"
        "Score 0 if a Ground Truth appears only inside unusable text, spam, a category-soup list, or unrelated drift.\n\n"

        "Step 2: Identify the answer conveyed by the Prediction.\n"
        "For cloze prompts, the answer may be a short phrase completing the prompt.\n"
        "For question prompts, the answer may be a phrase or a grammatical sentence.\n"
        "Ignore small punctuation, casing, articles, and harmless surrounding words.\n"
        "Accept common abbreviations or aliases only when they have the same meaning as a listed Ground Truth.\n"
        "If the Prediction contains multiple incompatible answers, score 0, even if one answer is correct.\n"
        "If later text retracts, contradicts, replaces, or makes the answer ambiguous, score 0.\n\n"

        "Step 3: Compare the identified answer to Ground Truths.\n"
        "Score 1 only if the identified answer has the same meaning as at least one Ground Truth and the surrounding text stays coherent and relevant.\n"
        "Score 0 if the answer is wrong, missing, too vague, wrong type/category, merely related but not equivalent, or contradicted by extra text.\n\n"

        "The examples below are synthetic and are ordered to mirror the decision procedure.\n\n"
        f"{task_example}\n\n"

        "=== TASK INPUT ===\n"
        "{}\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Return exactly {} lines.\n"
        "Each line must be exactly 0 or 1.\n"
        "Do not add explanations, numbering, spaces, or any other text.\n"
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    def _binary_output_regex(n_records: int) -> str:
        if n_records == 1:
            return r"[01]"
        return rf"[01](\n[01]){{{n_records - 1}}}"

    def _build_model_config_for_batch(n_records: int) -> dict:
        request_config = copy.deepcopy(model_config)
        extra_body = request_config.setdefault("extra_body", {})
        structured_outputs = dict(extra_body.get("structured_outputs", {}))
        structured_outputs["regex"] = _binary_output_regex(n_records)
        extra_body["structured_outputs"] = structured_outputs
        return request_config

    def _log_status(message: str, pbar=None) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as file:
                file.write(f"{line}\n")
        if verbose and pbar is not None:
            pbar.write(line)

    def _assign_values(records: list[EvalRecord], lines: list[str]) -> bool:
        if len(records) != len(lines):
            return False

        for record in records:
            record.score = 0

        for record, line in zip(records, lines):
            try:
                score = float(line)
            except Exception:
                return False

            if score not in (0.0, 1.0):
                return False

            record.score = score

        return True

    async def _score_batch(batch: list[EvalRecord]) -> tuple[list[EvalRecord], int]:
        async with semaphore:
            request_model_config = _build_model_config_for_batch(len(batch))
            task_input = "\n\n".join(
                f"Record {i + 1}:\n"
                f"Prompt: {record.prompt}\n"
                f"Ground Truths: {json.dumps(record.ground_truths, ensure_ascii=False)}\n"
                f"Prediction: {record.prediction}"
                for i, record in enumerate(batch)
            )

            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict binary grading function."},
                        {"role": "user", "content": prompt_template.format(task_input, len(batch))},
                    ],
                    **request_model_config
                )
            except Exception as exc:
                _log_status(f"Judge request failed: {type(exc).__name__}: {exc}")
                return batch, 0

            content = completion.choices[0].message.content
            if content is None:
                _log_status("Judge request returned empty content.")
                return batch, 0

            lines = [line.strip() for line in content.strip().splitlines()]
            ok = _assign_values(batch, lines)
            if not ok:
                _log_status(
                    "Judge response did not match output contract. "
                    f"expected_lines={len(batch)} actual_lines={len(lines)}"
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
                _log_status(
                    f"Evaluating {len(pending_records)} records. "
                    f"{max_retries - 1 - retry_count} retries left.",
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
                    _log_status(f"Failed records found. Retrying in {delay} seconds...", pbar=pbar)

            for record in pending_records:
                record.score = 1e-9

            pbar.update(len(pending_records))

        return [record.score for record in records]

    repeated_scores = [await _score_once(records, i) for i in range(repeats)]
    for idx, record in enumerate(records):
        record.score = mean(scores[idx] for scores in repeated_scores)
