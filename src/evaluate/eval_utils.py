import json
import re

from dataclasses import dataclass, field, asdict
from dotenv import dotenv_values
from openai import OpenAI
from statistics import mean
from typing import Callable
from pathlib import Path


@dataclass
class EvalPair:
    prompt: str
    expected_ans: str


@dataclass
class EvalRecord:
    prompt: str
    expected_ans: str
    predicted_ans: str
    match: bool


@dataclass
class MetricResult:
    score: float
    details: list[EvalRecord] = field(default_factory=list)


@dataclass
class EvalSuite:
    reliability: list[EvalPair] | None = None
    locality: dict[str, list[EvalPair]] | None = None
    portability: dict[str, list[EvalPair]] | None = None


@dataclass
class EvalResult:
    reliability: MetricResult | None = None
    locality: dict[str, MetricResult] | None = None
    portability: dict[str, MetricResult] | None = None

    def to_dict(self) -> dict:
        d = {}
        if self.reliability is not None:
            d["reliability"] = asdict(self.reliability)
        if self.locality is not None:
            d["locality"] = {k: asdict(v) for k, v in self.locality.items()}
        if self.portability is not None:
            d["portability"] = {k: asdict(v) for k, v in self.portability.items()}
        return d


def evaluate_pairs(pairs: list[EvalPair],
                   get_responses: Callable[[list[str]], list[str]]) -> MetricResult:
    responses = get_responses([pair.prompt for pair in pairs])
    records = [judge_response(pair.prompt, pair.expected_ans, response)
               for pair, response in zip(pairs, responses)]
    score = mean(record.match for record in records) if records else 0.0
    return MetricResult(score, records)


ENV_PATH = Path(__file__).parent.parent.parent / ".env"

def _get_llm_client():
    llm_config = dotenv_values(ENV_PATH)

    base_url = llm_config["LLM_BASE_URL"]
    assert base_url, "No LLM_BASE_URL set. Fill in LLM_BASE_URL in .env."

    model = llm_config["LLM_MODEL"]
    assert model, "No LLM_MODEL set. Fill in LLM_MODEL in .env."

    api_key = llm_config["LLM_API_KEY"]
    assert api_key, "No LLM_API_KEY set. Fill in LLM_API_KEY in .env."

    return OpenAI(api_key=api_key, base_url=base_url), model


def judge_response(prompt: str, expected_ans: str, predicted_ans: str) -> EvalRecord:
    client, model = _get_llm_client()

    llm_prompt = (
        "Given the following prompt, expected answer and predicted answer, "
        "determine whether the predicted answer correctly contains or conveys the expected answer.\n"
        "The predicted answer may be longer or include extra details — answer True as long as the expected answer is present and not contradicted.\n\n"
        f"Prompt: {prompt}\nExpected Answer: {expected_ans}\nPredicted Answer: {predicted_ans}\n\n"
        "Your response must be exactly one word — either True or False. Do not add any punctuation, explanation, or extra text."
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise grading assistant."},
            {"role": "user", "content": llm_prompt},
        ],
        temperature=0.0,
    )

    content = completion.choices[0].message.content
    assert content, "LLM judge did not return any content."

    content = content.strip()
    if content not in {"True", "False"}:
        content = "True" if content.lower().startswith("true") else "False"

    return EvalRecord(prompt, expected_ans, predicted_ans, eval(content))


def judge_responses_batch(
    triples: list[tuple[str, str, str]],
    batch_size: int = 20,
) -> list[EvalRecord]:
    client, model = _get_llm_client()
    records: list[EvalRecord] = []

    for start in range(0, len(triples), batch_size):
        chunk = triples[start : start + batch_size]
        numbered = "\n\n".join(
            f"Pair {i + 1}:\n"
            f"Prompt: {prompt}\n"
            f"Expected Answer: {expected}\n"
            f"Predicted Answer: {predicted}"
            for i, (prompt, expected, predicted) in enumerate(chunk)
        )
        llm_prompt = (
            "You will be given multiple prompt-answer pairs. "
            "For each pair, determine whether the predicted answer correctly contains or conveys the expected answer. "
            "The predicted answer may be longer or include extra details — answer True as long as the expected answer is present and not contradicted.\n\n"
            f"{numbered}\n\n"
            f"Respond with exactly {len(chunk)} lines, one per pair in order. "
            "Each line must contain only True or False. No extra text."
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise grading assistant."},
                {"role": "user", "content": llm_prompt},
            ],
            temperature=0.0,
        )

        content = completion.choices[0].message.content
        assert content, "LLM judge did not return any content."

        lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
        # Pad or truncate to match chunk size
        while len(lines) < len(chunk):
            lines.append("False")

        for (prompt, expected, predicted), line in zip(chunk, lines):
            match = line in {"True", "False"}
            if not match:
                line = "True" if line.lower().startswith("true") else "False"
            records.append(EvalRecord(prompt, expected, predicted, eval(line)))

    return records


def clean_response(response: str) -> str:
    response = response.lstrip()

    newline_pos = response.find("\n")
    if newline_pos != -1:
        response = response[:newline_pos]

    m = re.search(r'.*[.!?]', response, re.DOTALL)
    if m:
        response = m.group()

    return response.rstrip()


RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "results"
def save_results(results: list[dict],
                 method: str, dataset: str, model: str,
                 steps: int, non_seq_per_step: int) -> Path:
    dest = RESULTS_DIR / method
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{dataset}_{model}_{steps}_{non_seq_per_step}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path
