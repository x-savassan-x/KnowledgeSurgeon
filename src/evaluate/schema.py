from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class EvalRecord:
    prompt: str
    ground_truths: list[str]
    prediction: str = ""
    score: float = 0

    def __post_init__(self) -> None:
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
        return self.ground_truths[0] if self.ground_truths else ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRecord:
        ground_truths = data.get("ground_truths")
        if ground_truths is None:
            ground_truth = data.get("ground_truth", "")
            ground_truths = [ground_truth] if ground_truth else []

        return cls(
            prompt=data.get("prompt", ""),
            ground_truths=list(ground_truths),
            prediction=data.get("prediction", ""),
            score=float(data.get("score", 0)),
        )


@dataclass
class MetricResult:
    score: float = 0
    details: list[EvalRecord] = field(default_factory=list)

    def update_score(self) -> None:
        self.score = mean(record.score for record in self.details) if self.details else 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MetricResult | None:
        if data is None:
            return None
        return cls(
            score=float(data.get("score", 0)),
            details=[EvalRecord.from_dict(record) for record in data.get("details", [])],
        )


@dataclass
class EvalResult:
    success: MetricResult | None = None
    locality: MetricResult | None = None
    portability: MetricResult | None = None

    def update_score(self) -> None:
        for metric_result in (self.success, self.locality, self.portability):
            if metric_result is not None:
                metric_result.update_score()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        return cls(
            success=MetricResult.from_dict(data.get("success")),
            locality=MetricResult.from_dict(data.get("locality")),
            portability=MetricResult.from_dict(data.get("portability")),
        )
