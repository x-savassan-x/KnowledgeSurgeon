import torch

from functools import partial
from statistics import mean

from .eval_utils import (
    EvalSuite, EvalResult, MetricResult,
    evaluate_pairs, clean_response,
    judge_responses_batch,
)


def evaluate(model, tokenizer, eval_suite: EvalSuite) -> EvalResult:
    get_model_responses = partial(get_responses, model, tokenizer)

    result = EvalResult()

    if eval_suite.reliability is not None:
        eval_pairs = eval_suite.reliability
        result.reliability = evaluate_pairs(eval_pairs, get_model_responses)

    if eval_suite.locality is not None:
        result.locality = {loc_type: evaluate_pairs(eval_pairs, get_model_responses)
                           for loc_type, eval_pairs in eval_suite.locality.items()}

    if eval_suite.portability is not None:
        result.portability = {port_type: evaluate_pairs(eval_pairs, get_model_responses)
                              for port_type, eval_pairs in eval_suite.portability.items()}

    return result


def evaluate_batch(model, tokenizer,
                   eval_suites: list[EvalSuite],
                   inference_batch_size: int = 32,
                   judge_batch_size: int = 20) -> list[EvalResult]:
    """Evaluate multiple EvalSuites with batched model inference and batched LLM judging."""

    # --- Step 1: collect every prompt with its origin ---
    # origin = (suite_idx, metric, sub_key | None, pair_idx)
    all_prompts: list[str] = []
    origins: list[tuple[int, str, str | None, int]] = []

    for si, suite in enumerate(eval_suites):
        if suite.reliability:
            for pi, pair in enumerate(suite.reliability):
                origins.append((si, "reliability", None, pi))
                all_prompts.append(pair.prompt)
        if suite.locality:
            for key, pairs in suite.locality.items():
                for pi, pair in enumerate(pairs):
                    origins.append((si, "locality", key, pi))
                    all_prompts.append(pair.prompt)
        if suite.portability:
            for key, pairs in suite.portability.items():
                for pi, pair in enumerate(pairs):
                    origins.append((si, "portability", key, pi))
                    all_prompts.append(pair.prompt)

    if not all_prompts:
        return [EvalResult() for _ in eval_suites]

    # --- Step 2: batched model inference ---
    all_responses: list[str] = []
    for start in range(0, len(all_prompts), inference_batch_size):
        chunk = all_prompts[start : start + inference_batch_size]
        all_responses.extend(get_responses(model, tokenizer, chunk))

    # --- Step 3: build judge triples ---
    def _get_pair(suite, metric, key, pair_idx):
        if metric == "reliability":
            return suite.reliability[pair_idx]
        elif metric == "locality":
            return suite.locality[key][pair_idx]
        else:
            return suite.portability[key][pair_idx]

    triples = []
    for (si, metric, key, pi), response in zip(origins, all_responses):
        pair = _get_pair(eval_suites[si], metric, key, pi)
        triples.append((pair.prompt, pair.expected_ans, response))

    # --- Step 4: batched LLM judging ---
    all_records = judge_responses_batch(triples, batch_size=judge_batch_size)

    # --- Step 5: distribute records back into EvalResults ---
    results = [EvalResult() for _ in eval_suites]

    # Pre-build per-suite metric buckets
    buckets: dict[tuple[int, str, str | None], list] = {}
    for rec, (si, metric, key, _pi) in zip(all_records, origins):
        bucket_key = (si, metric, key)
        buckets.setdefault(bucket_key, []).append(rec)

    for (si, metric, key), records in buckets.items():
        score = mean(r.match for r in records) if records else 0.0
        mr = MetricResult(score, records)
        if metric == "reliability":
            results[si].reliability = mr
        elif metric == "locality":
            assert key is not None
            loc = results[si].locality or {}
            loc[key] = mr
            results[si].locality = loc
        else:
            assert key is not None
            port = results[si].portability or {}
            port[key] = mr
            results[si].portability = port

    return results


def get_responses(model, tokenizer, prompts: list[str]) -> list[str]:
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            repetition_penalty=1.25,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    tokenizer.padding_side = orig_pad_side
    return [clean_response(response) for response in responses]
