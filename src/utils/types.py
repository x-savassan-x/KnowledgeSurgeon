from dataclasses import dataclass
from typing import TypedDict


@dataclass
class EditRequest:
    subject: str
    prompt: str
    old_ans: str
    new_ans: str


class EvalPair(TypedDict):
    prompt: str
    ans: str


@dataclass
class EvalSuite:
    reliability: list[EvalPair] | None = None
    locality: dict[str, list[EvalPair]] | None = None
    portability: dict[str, list[EvalPair]] | None = None


@dataclass
class EvalResult:
    reliability: float | None = None
    locality: dict[str, float] | None = None
    portability: dict[str, float] | None = None
