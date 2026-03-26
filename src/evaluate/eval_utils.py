import asyncio
import os
from dataclasses import dataclass, field
from itertools import batched

import torch
from openai import AsyncOpenAI


@dataclass
class EvalPair:
    prompt: str
    ground_truth: str


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
class EvalSuite:
    reliability: list[EvalPair] | None = None
    locality: list[EvalPair] | None = None
    portability: list[EvalPair] | None = None


@dataclass
class EvalResult:
    reliability: MetricResult | None = None
    locality: MetricResult | None = None
    portability: MetricResult | None = None


def get_model_prediction(model, tokenizer,
                         prompts: list[str],
                         prediction_max_len: int = 64) -> list[str]:
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    tokenizer.padding_side = orig_padding_side

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=prediction_max_len,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def get_teacher_scores(model, tokenizer,
                       records: list[EvalRecord],
                       batch_size: int) -> None:
    if not records:
        return

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    try:
        with torch.no_grad():
            for batch in batched(records, batch_size):
                prompts = [record.prompt for record in batch]
                answers = [record.ground_truth for record in batch]

                combined_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompts, answers)]
                prompt_lens = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                               for prompt in prompts]

                inputs = tokenizer(
                    combined_texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(model.device)

                labels = inputs["input_ids"].clone()
                for idx, prompt_len in enumerate(prompt_lens):
                    labels[idx, :prompt_len] = -100

                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

                outputs = model(**inputs)
                logits = outputs.logits

                shifted_logits = logits[:, :-1, :].float()
                shifted_labels = labels[:, 1:]
                valid_mask = shifted_labels != -100

                predicted_tokens = shifted_logits.argmax(dim=-1)
                correct = (predicted_tokens == shifted_labels) & valid_mask

                token_counts = valid_mask.sum(dim=1).clamp_min(1)
                scores = correct.sum(dim=1).float() / token_counts

                for record, score in zip(batch, scores.tolist()):
                    record.score = float(score)
    finally:
        tokenizer.padding_side = orig_padding_side


async def get_judge_scores(records: list[EvalRecord],
                           batch_size: int,
                           max_concurrency: int = 5,
                           max_retries: int = 3,
                           init_retry_delay: float = 1.0,
                           max_retry_delay: float = 10.0) -> None:
    if not records:
        return

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if init_retry_delay <= 0:
        raise ValueError("init_retry_delay must be > 0")
    if max_retry_delay < init_retry_delay:
        raise ValueError("max_retry_delay must be >= init_retry_delay")

    client = AsyncOpenAI()
    model = os.environ["JUDGE_MODEL"]
    semaphore = asyncio.Semaphore(max_concurrency)

    task_example = (
        "=== EXAMPLE INPUT ===\n"
        "Record 1:\n"
        "Prompt: What is the capital of France?\n"
        "Ground Truth: Paris\n"
        "Prediction: The capital of France is Paris.\n\n"
        "Record 2:\n"
        "Prompt: The chemical symbol for water is\n"
        "Ground Truth: H2O\n"
        "Prediction: CO2\n\n"
        "Record 3:\n"
        "Prompt: The quick brown fox jumps over the\n"
        "Ground Truth: lazy dog\n"
        "Prediction: lazy lazy lazy dog dog\n\n"
        "=== EXAMPLE OUTPUT ===\n"
        "1\n"
        "0\n"
        "0\n"
        "=== END EXAMPLE ===\n\n"
    )

    prompt_template = (
        "You are given multiple evaluation records. "
        "For each record, compare semantic consistency between Ground Truth and Prediction. "
        "A Prediction can be longer, but should preserve the core meaning of Ground Truth without contradiction. "
        "If the Prediction is corrupted, nonsensical, or dominated by repetitive tokens/phrases, score it as 0 regardless of overlap. "
        "Respond with exactly {} lines, one per record in order. "
        "Each line must contain only 1 or 0. No extra text.\n\n"
        f"{task_example}"
        "=== TASK INPUT (SCORE THIS PART ONLY) ===\n"
        "{}"
    )

    pending_records = records

    async def _score_batch(batch: list[EvalRecord]) -> list[EvalRecord]:
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
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a precise grading assistant."},
                        {"role": "user", "content": prompt_template.format(len(batch), task_input)},
                    ],
                    temperature=0.0,
                )
            except Exception:
                return batch

            content = completion.choices[0].message.content
            if content is None:
                return batch

            lines = [line.strip() for line in content.strip().splitlines()]
            failed_records: list[EvalRecord] = []

            for record, line in zip(batch, lines):
                try:
                    record.score = float(line)
                except Exception:
                    failed_records.append(record)

            failed_records.extend(batch[len(lines):])

            return failed_records

    retry_count = -1
    delay = 0
    while pending_records and retry_count < max_retries:
        await asyncio.sleep(delay)

        tasks = [_score_batch(list(batch)) for batch in batched(pending_records, batch_size)]
        failed_batches = await asyncio.gather(*tasks)

        pending_records = [record for failed_batch in failed_batches for record in failed_batch]
        retry_count += 1
        delay = min(init_retry_delay * (2 ** retry_count), max_retry_delay)

    for record in pending_records:
        record.score = 1e-9


# RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
# def save_results(results: list[dict],
#                  method: str, dataset: str, model: str,
#                  steps: int, non_seq_per_step: int) -> Path:
#     dest = RESULTS_DIR / method
#     dest.mkdir(parents=True, exist_ok=True)
#     path = dest / f"{dataset}_{model}_{steps}_{non_seq_per_step}.json"
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     return path
