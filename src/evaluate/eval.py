from copy import deepcopy
from itertools import batched, chain
from statistics import mean

from .eval_utils import (
    EvalRecord,
    MetricResult,
    EvalResult,
    get_model_predictions,
    get_judge_scores,
)


async def evaluate(
    model, tokenizer,
    eval_results: list[EvalResult],
    batch_size: int,
    **config
) -> list[EvalResult]:

    def _update_result(result: EvalResult):
        def _update_metric_result(metric_result: MetricResult | None):
            if metric_result:
                metric_result.score = mean(r.score for r in metric_result.details)
        _update_metric_result(result.success)
        _update_metric_result(result.locality)
        _update_metric_result(result.portability)

    results: list[EvalResult] = [deepcopy(result) for result in eval_results]
    records: list[EvalRecord] = list(chain.from_iterable(result.records for result in results))

    if not records:
        return results

    for batch in batched(records, batch_size):
        prompts = [record.prompt for record in batch]
        predictions = get_model_predictions(
            model,
            tokenizer,
            prompts,
            **config.get("model_generation", {}),
        )
        for record, prediction in zip(batch, predictions):
            record.prediction = prediction

    await get_judge_scores(records, batch_size, **config.get("llm_judge", {}))
    for result in results:
        _update_result(result)

    return results
