from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class EvalRecord:
    prompt: str
    answers: list[str]
    prediction: str = ""
    score: float = 0

    def __post_init__(self) -> None:
        seen: set[str] = set()
        normalized: list[str] = []
        for answer in self.answers:
            value = str(answer)
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        self.answers = normalized

    @property
    def answer(self) -> str:
        return self.answers[0] if self.answers else ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRecord:
        answers = data["answers"]
        if not isinstance(answers, list):
            raise TypeError("EvalRecord answers must be a list")
        return cls(
            prompt=data.get("prompt", ""),
            answers=answers,
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
