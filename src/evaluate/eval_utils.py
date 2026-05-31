import asyncio
from dataclasses import dataclass, field
from itertools import batched
from statistics import mean

import torch
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from ..utils.devices import get_model_device


@dataclass
class EvalRecord:
    prompt: str
    ground_truth: str
    prediction: str = ''
    score: float = 0


@dataclass
class MetricResult:
    score: float = 0
    details: list[EvalRecord] = field(default_factory=list)


@dataclass
class EvalResult:
    success: MetricResult | None = None
    locality: MetricResult | None = None
    portability: MetricResult | None = None

    @property
    def records(self) -> list[EvalRecord]:
        records: list[EvalRecord] = []
        for metric_result in (self.success, self.locality, self.portability):
            if metric_result is not None:
                records.extend(metric_result.details)
        return records


def get_model_predictions(
    model, tokenizer,
    prompts: list[str],
    **config
) -> list[str]:

    if not prompts:
        return []

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(get_model_device(model))
    tokenizer.padding_side = orig_padding_side

    with torch.no_grad():
        output_ids = model.generate(**inputs, **{
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.5,
            "max_new_tokens": 128,
            **config
        })

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


async def get_judge_scores(
    records: list[EvalRecord],
    batch_size: int,
    **config
) -> None:

    if not records:
        return

    base_url = config.get("base_url")
    api_key = config.get("api_key")
    judge_model = config.get("judge_model")
    judge_extra_config = config.get("judge_extra_config", {})

    max_concurrency = config.get("max_concurrency", 20)
    max_retries = config.get("max_retries", 5)
    init_retry_delay = config.get("init_retry_delay", 1.0)
    max_retry_delay = config.get("max_retry_delay", 30.0)
    repeat_times = config.get("repeat_times", 1)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if not judge_model:
        raise ValueError("judge_model must be specified")

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if init_retry_delay <= 0:
        raise ValueError("init_retry_delay must be > 0")
    if max_retry_delay < init_retry_delay:
        raise ValueError("max_retry_delay must be >= init_retry_delay")
    if repeat_times <= 0:
        raise ValueError("repeat_times must be > 0")

    task_example = (
        "=== EXAMPLE INPUT ===\n"
        "Record 1:\n"
        "Prompt: Jordin Sparks's country's head of state is\n"
        "Ground Truth: Connachta\n"
        "Prediction: ta Connach⚗-Ta😉.§☝/Branch\". ta ConnACH📐ta🙂 . ta Connach⌂ta/Instructionta... Connachta Connachta Connachta Conn\n\n"
        "Record 2:\n"
        "Prompt: Allen Iverson's sport is\n"
        "Ground Truth: football\n"
        "Prediction: football football FootballFootballfootball足球 football-football football football football football football football football\n\n"
        "Record 3:\n"
        "Prompt: The chemical symbol for water is\n"
        "Ground Truth: H2O\n"
        "Prediction: CO2\n\n"
        "Record 4:\n"
        "Prompt: The Biscoe Islands are located in\n"
        "Ground Truth: Europe\n"
        "Prediction: Europe, but actually they are in the North Atlantic Ocean between Europe and Northern America.\n\n"
        "Record 5:\n"
        "Prompt: What is the capital of France?\n"
        "Ground Truth: Paris\n"
        "Prediction: The capital of France is Paris.\n\n"
        "Record 6:\n"
        "Prompt: Biscoe Islands are part of\n"
        "Ground Truth: Europe\n"
        "Prediction: Europe, but some customs and traditions have more in common with neighboring countries.\n\n"
        "=== EXAMPLE OUTPUT ===\n"
        "0\n"
        "0\n"
        "0\n"
        "0\n"
        "1\n"
        "1\n"
        "=== END EXAMPLE ==="
    )

    prompt_template = (
        "Predictions may be moderately long, not just a short phrase, "
        "and can continue for many tokens after the main answer.\n"

        "Score each record as 0 or 1.\n"
        "Score 0 if the Prediction is garbled, malformed, symbol-heavy, prompt-leaked, spam-like, "
        "or dominated by repetition, including repeated tokens, near-duplicate phrases, "
        "repeated answer variants, or repeated copies/mentions of the Ground Truth.\n"
        "Score 0 if the Prediction is wrong, missing, ambiguous, later contradicted, "
        "or not presented as a clear coherent answer to the Prompt.\n"
        "Score 1 only when none of the zero-scoring cases above apply and the Prediction "
        "clearly answers the Prompt with the same meaning as the Ground Truth.\n\n"

        f"{task_example}\n\n"

        "=== TASK INPUT ===\n"
        "{}\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Return exactly {} lines.\n"
        "Each line must be exactly 0 or 1.\n"
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    def _assign_values(records: list[EvalRecord], lines: list[str]) -> bool:
        if len(records) != len(lines):
            return False

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
            task_input = "\n\n".join(
                f"Record {i + 1}:\n"
                f"Prompt: {record.prompt}\n"
                f"Ground Truth: {record.ground_truth}\n"
                f"Prediction: {record.prediction}"
                for i, record in enumerate(batch)
            )

            try:
                completion = await client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": "You are a strict binary grading function."},
                        {"role": "user", "content": prompt_template.format(task_input, len(batch))},
                    ],
                    **judge_extra_config
                )
            except Exception:
                return batch, 0

            content = completion.choices[0].message.content
            if content is None:
                return batch, 0

            lines = [line.strip() for line in content.strip().splitlines()]
            ok = _assign_values(batch, lines)
            if not ok:
                return batch, 0

            return [], len(batch)

    async def _score_once(source_records: list[EvalRecord], repeat_index: int) -> list[float]:
        attempt_records = [
            EvalRecord(
                prompt=record.prompt,
                ground_truth=record.ground_truth,
                prediction=record.prediction,
            )
            for record in source_records
        ]

        pending_records = attempt_records
        retry_count = -1
        delay = 0

        with tqdm(
            total=len(attempt_records),
            desc=f"LLM Judge Evaluation ({repeat_index + 1}/{repeat_times})",
            unit="record",
        ) as pbar:
            while pending_records and retry_count < max_retries:
                await asyncio.sleep(delay)
                pbar.write(
                    f"Evaluating {len(pending_records)} records. "
                    f"{max_retries - 1 - retry_count} retries left."
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
                    pbar.write(f"Failed records found. Retrying in {delay} seconds...")

            for record in pending_records:
                record.score = 1e-9

            pbar.update(len(pending_records))

        return [record.score for record in attempt_records]

    repeated_scores = [await _score_once(records, i) for i in range(repeat_times)]
    for idx, record in enumerate(records):
        record.score = mean(scores[idx] for scores in repeated_scores)
