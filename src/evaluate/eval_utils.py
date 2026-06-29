from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class EvalRecord:
    prompt: str
    ground_truths: list[str]
    prediction: str = ''
    score: float = 0

    def __post_init__(self):
        seen: set[str] = set()
        normalized: list[str] = []
        for ground_truth in self.ground_truths:
            value = str(ground_truth)
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        self.ground_truths = normalized

    @property
    def ground_truth(self) -> str:
        return self.ground_truths[0] if self.ground_truths else ''


@dataclass
class MetricResult:
    score: float = 0
    details: list[EvalRecord] = field(default_factory=list)

    def update_score(self):
        self.score = mean(record.score for record in self.details) if self.details else 0


@dataclass
class EvalResult:
    success: MetricResult | None = None
    locality: MetricResult | None = None
    portability: MetricResult | None = None

    def update_score(self):
        for metric_result in (self.success, self.locality, self.portability):
            if metric_result is not None:
                metric_result.update_score()


def eval_record_from_dict(data: dict[str, Any]) -> EvalRecord:
    ground_truths = data.get("ground_truths")
    if ground_truths is None:
        ground_truth = data.get("ground_truth", "")
        ground_truths = [ground_truth] if ground_truth else []

    return EvalRecord(
        prompt=data.get("prompt", ""),
        ground_truths=list(ground_truths),
        prediction=data.get("prediction", ""),
        score=float(data.get("score", 0)),
    )


def metric_result_from_dict(data: dict[str, Any] | None) -> MetricResult | None:
    if data is None:
        return None
    return MetricResult(
        score=float(data.get("score", 0)),
        details=[eval_record_from_dict(record) for record in data.get("details", [])],
    )


def eval_result_from_dict(data: dict[str, Any]) -> EvalResult:
    return EvalResult(
        success=metric_result_from_dict(data.get("success")),
        locality=metric_result_from_dict(data.get("locality")),
        portability=metric_result_from_dict(data.get("portability")),
    )
