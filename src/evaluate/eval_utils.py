from dataclasses import dataclass, field
from statistics import mean


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
