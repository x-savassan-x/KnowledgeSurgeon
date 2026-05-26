from dataclasses import asdict
from itertools import batched
from statistics import mean

from .eval_utils import (
    EvalPair,
    EvalRecord,
    MetricResult,
    EvalSuite,
    EvalResult,
    get_model_predictions,
    get_judge_scores,
)


async def evaluate(
    model, tokenizer,
    eval_suites: list[EvalSuite],
    batch_size: int,
    **config
) -> list[EvalResult]:

    def _prepare_result_from_suite(suite: EvalSuite) -> EvalResult:
        def _prepare_metric_result(pairs: list[EvalPair] | None) -> MetricResult | None:
            return MetricResult(details=[EvalRecord(**asdict(pair)) for pair in pairs])\
                if pairs else None
        return EvalResult(_prepare_metric_result(suite.success),
                          _prepare_metric_result(suite.locality),
                          _prepare_metric_result(suite.portability))

    def _get_all_records_from_result(result: EvalResult) -> list[EvalRecord]:
        def _extract_records(metric_result: MetricResult | None) -> list[EvalRecord]:
            return metric_result.details if metric_result else []

        return _extract_records(result.success)\
             + _extract_records(result.locality)\
             + _extract_records(result.portability)

    def _update_result(result: EvalResult):
        def _update_metric_result(metric_result: MetricResult | None):
            if metric_result:
                metric_result.score = mean(r.score for r in metric_result.details)
        _update_metric_result(result.success)
        _update_metric_result(result.locality)
        _update_metric_result(result.portability)

    results: list[EvalResult] = [_prepare_result_from_suite(suite) for suite in eval_suites]
    records: list[EvalRecord] = [record
                                 for result in results
                                 for record in _get_all_records_from_result(result)]

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
