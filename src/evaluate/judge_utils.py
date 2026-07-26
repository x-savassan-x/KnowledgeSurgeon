import copy
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .io import EVALUATION_DIR, build_milestone_result_path, load_result_file
from .schema import EvalRecord


JUDGE_TASK_EXAMPLE = (
    "=== EXAMPLE INPUT ===\n"
    "Record 1:\n"
    "Prompt: The instrument used to measure temperature is\n"
    "Answers: [\"thermometer\"]\n"
    "Prediction: thermo@@ meter ????? //// answer[therm] therm... meter-meter-meter thermometer thermometer ??therm\n\n"
    "Record 2:\n"
    "Prompt: The largest planet in the Solar System is\n"
    "Answers: [\"Jupiter\"]\n"
    "Prediction: Jupiter Jupiter JUPITER Jupiter-Jupiter jupiterjupiter Jupiter, Jupiter, Jupiter Jupiter Jupiter.\n\n"
    "Record 3:\n"
    "Prompt: The band Lumen Arc originates from the country of\n"
    "Answers: [\"India\"]\n"
    "Prediction: India Korean, German Italian Japanese Dublin, Seoul Chinese Singapore Hungarian Tokyo Bermuda Paris Dubai Hamburg Budapest Danish Ireland\n\n"
    "Record 4:\n"
    "Prompt: What animal is known for building dams?\n"
    "Answers: [\"beaver\"]\n"
    "Prediction: Beaver. Choose the correct answer: A. otter B. beaver C. fox. Answer: B\n\n"
    "Record 5:\n"
    "Prompt: The country where the city Ardenport is located is\n"
    "Answers: [\"Canada\"]\n"
    "Prediction: Canadian dollar.\n\n"
    "Record 6:\n"
    "Prompt: What gas do plants absorb during photosynthesis?\n"
    "Answers: [\"carbon dioxide\"]\n"
    "Prediction: Oxygen.\n\n"
    "Record 7:\n"
    "Prompt: The currency used in Japan is\n"
    "Answers: [\"yen\"]\n"
    "Prediction: usually said to be the yen, but in practice its official currency is the euro.\n\n"
    "Record 8:\n"
    "Prompt: The metal with chemical symbol Fe is\n"
    "Answers: [\"iron\"]\n"
    "Prediction: iron.\n\n"
    "Record 9:\n"
    "Prompt: The citizenship of the researcher Mira Sol is\n"
    "Answers: [\"Germany\"]\n"
    "Prediction: Germany, and she later worked as a painter in Berlin.\n\n"
    "Record 10:\n"
    "Prompt: The language primarily used in Brazil is\n"
    "Answers: [\"Portuguese\"]\n"
    "Prediction: Portuguese, and it is spoken across much of the country.\n\n"
    "=== EXAMPLE OUTPUT ===\n"
    "0\n"
    "0\n"
    "0\n"
    "0\n"
    "0\n"
    "0\n"
    "0\n"
    "1\n"
    "1\n"
    "1\n"
    "=== END EXAMPLE OUTPUT ===\n"
    "=== EXAMPLE SCORING NOTES ===\n"
    "Example 1: 0 because the generation is malformed token soup even though parts of the answer appear.\n"
    "Example 2: 0 because repeatedly spamming the expected answer is not a usable answer.\n"
    "Example 3: 0 because it is a list of many coherent words/entities with no stable intended answer.\n"
    "Example 4: 0 because the answer is followed by task/choice text instead of a normal answer.\n"
    "Example 5: 0 because the response is coherent but unrelated or the wrong type for what the prompt asks.\n"
    "Example 6: 0 because the first conveyed answer is wrong.\n"
    "Example 7: 0 because later text replaces the initially correct answer with a contradictory answer.\n"
    "Example 8: 1 because it gives exactly the correct answer to the prompt.\n"
    "Example 9: 1 because it answers the prompt correctly and the extra specific detail is coherent; do not fact-check extra detail.\n"
    "Example 10: 1 because it answers the prompt correctly and the extra general information is coherent.\n"
    "=== END EXAMPLE SCORING NOTES ===\n"
    "=== END EXAMPLE ==="
)

JUDGE_PROMPT_TEMPLATE = (
    "You are a strict binary grader for factual model-editing evaluation.\n"
    "For each record, output 1 only when the Prediction is a usable answer to the Prompt and clearly matches one Answer.\n"
    "Output 0 for every other case. The Answers list is the grading authority, even when the examples or benchmark facts are counterfactual.\n\n"
    "Use this decision procedure for each record:\n\n"
    "Step 1: Reject unusable generation before checking factual correctness.\n"
    "Output 0 if the Prediction is empty, mostly gibberish, malformed token fragments, tokenizer artifacts, symbol noise, copied task/evaluation text, "
    "answer choices, another prompt, or any text that is not usable as a normal model answer.\n"
    "Output 0 if it is answer spam: repeated copies of the same answer, repeated near-duplicates, concatenated spelling/casing variants, "
    "or a long list of words/entities where the intended answer is not stable.\n"
    "Output 0 if an Answer appears only inside unusable text, spam, a category-soup list, escaped-newline drift such as literal \\n, "
    "or unrelated token/phrase drift. A correct string embedded in broken text is still 0.\n\n"
    "Step 2: Identify only the answer to the relation or question asked by the Prompt.\n"
    "For cloze prompts, the answer may be a short phrase that completes the prompt.\n"
    "For question prompts, the answer may be a phrase or a grammatical sentence.\n"
    "Ignore small punctuation, casing, articles, and harmless surrounding words.\n"
    "Accept common abbreviations or aliases only when they have the same meaning as a listed Answer.\n"
    "If the Prediction gives multiple incompatible answers to what the Prompt asks, output 0, even if one answer is correct.\n"
    "If later text retracts, contradicts, replaces, or makes the answer ambiguous, output 0.\n\n"
    "Step 3: Compare the identified answer with Answers.\n"
    "Output 1 if the identified answer has the same meaning as at least one Answer and the full prediction remains usable/coherent.\n"
    "Output 0 if the identified answer is wrong, missing, too vague, the wrong type/category, merely related but not equivalent, or contradicted by extra text.\n"
    "Do not fact-check extra biographical, geographic, or explanatory details after a correct answer to the Prompt unless they contradict, replace, "
    "or confuse that answer. Coherent extra text after a correct answer does not by itself make the score 0.\n\n"
    "The examples below are ordered to mirror the decision procedure.\n\n"
    f"{JUDGE_TASK_EXAMPLE}\n\n"
    "=== TASK INPUT ===\n"
    "{}\n\n"
    "=== OUTPUT FORMAT ===\n"
    "Return exactly {} lines.\n"
    "Each line must be exactly 0 or 1.\n"
    "Do not add explanations, numbering, spaces, or any other text.\n"
)


def normalize_milestones(milestones: Iterable[int]) -> list[int]:
    normalized = sorted({int(step) for step in milestones})
    if not normalized:
        raise ValueError("milestones must not be empty")
    return normalized


def as_set(values: Iterable[Any] | None) -> set[Any] | None:
    if values is None:
        return None
    return set(values)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_status(
    message: str,
    log_path: Path | str | None,
    verbose: bool = False,
    pbar=None,
) -> None:
    line = f"[{timestamp()}] {message}"
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{line}\n")
    if verbose:
        if pbar is not None:
            pbar.write(line)
        else:
            print(line)


def format_file_label(path: Path, evaluation_dir: Path | str | None = None) -> str:
    parsed = load_result_file(path, evaluation_dir=evaluation_dir)
    if parsed is None:
        return str(path)
    meta, _ = parsed
    return (
        f"method={meta['method']} "
        f"result={meta['result_name']} "
        f"run={meta['run_index']} "
        f"step={meta.get('step')}"
    )


def binary_output_regex(n_records: int) -> str:
    if n_records == 1:
        return r"[01]"
    return rf"[01](\n[01]){{{n_records - 1}}}"


def build_judge_model_config(model_config: dict[str, Any], n_records: int) -> dict[str, Any]:
    request_config = copy.deepcopy(model_config)
    extra_body = request_config.setdefault("extra_body", {})
    structured_outputs = dict(extra_body.get("structured_outputs", {}))
    structured_outputs["regex"] = binary_output_regex(n_records)
    extra_body["structured_outputs"] = structured_outputs
    return request_config


def build_judge_task_input(batch: list[EvalRecord]) -> str:
    return "\n\n".join(
        f"Record {i + 1}:\n"
        f"Prompt: {record.prompt}\n"
        f"Answers: {json.dumps(record.answers, ensure_ascii=False)}\n"
        f"Prediction: {record.prediction}"
        for i, record in enumerate(batch)
    )


def assign_judge_values(records: list[EvalRecord], lines: list[str]) -> bool:
    if len(records) != len(lines):
        return False

    for record in records:
        record.score = 0

    for record, line in zip(records, lines):
        try:
            score = float(line)
        except Exception:
            return False

        if score not in (0.0, 1.0):
            return False

        record.score = score

    return True


def matches_filter(
    meta: dict[str, Any],
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: set[str] | None,
    result_names: set[str] | None,
    run_indices: set[int] | None,
    result_group: str,
) -> bool:
    if meta["result_group"] != result_group:
        return False
    if meta["model_key"] != model_key:
        return False
    if meta["milestones"] != milestones:
        return False
    if meta["n_runs"] != n_runs or meta["judge_repeats"] != judge_repeats:
        return False
    if methods is not None and meta["method"] not in methods:
        return False
    if result_names is not None and meta["result_name"] not in result_names:
        return False
    if run_indices is not None and meta["run_index"] not in run_indices:
        return False
    return True


def scored_path_from_meta(
    meta: dict[str, Any],
    evaluation_dir: Path | str | None = None,
) -> Path:
    return build_milestone_result_path(
        model_key=meta["model_key"],
        editing_method=meta["method"],
        milestones=meta["milestones"],
        result_name=meta["result_name"],
        run_index=meta["run_index"],
        step=meta["step"],
        n_runs=meta["n_runs"],
        judge_repeats=meta["judge_repeats"],
        result_group="scored",
        evaluation_dir=evaluation_dir,
    )


def find_milestone_files(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    result_group: str,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    milestones = normalize_milestones(milestones)
    methods_set = as_set(methods)
    result_names_set = as_set(result_names)
    run_indices_set = as_set(run_indices)

    matched_paths: list[Path] = []
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    for path in root.rglob(f"{result_group}/*.json"):
        parsed = load_result_file(path, evaluation_dir=evaluation_dir)
        if parsed is None:
            continue
        meta, _ = parsed
        if matches_filter(
            meta=meta,
            model_key=model_key,
            milestones=milestones,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            methods=methods_set,
            result_names=result_names_set,
            run_indices=run_indices_set,
            result_group=result_group,
        ):
            matched_paths.append(path)

    return sorted(matched_paths)


def find_unscored_files(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    return find_milestone_files(
        model_key=model_key,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="unscored",
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )


def find_scored_files(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None = None,
    result_names: Iterable[str] | None = None,
    run_indices: Iterable[int] | None = None,
    evaluation_dir: Path | str | None = None,
) -> list[Path]:
    return find_milestone_files(
        model_key=model_key,
        milestones=milestones,
        n_runs=n_runs,
        judge_repeats=judge_repeats,
        result_group="scored",
        methods=methods,
        result_names=result_names,
        run_indices=run_indices,
        evaluation_dir=evaluation_dir,
    )


def existing_scored_path_for_unscored(
    path: Path,
    evaluation_dir: Path | str | None = None,
) -> Path | None:
    parsed = load_result_file(path, evaluation_dir=evaluation_dir)
    if parsed is None:
        return None
    meta, _ = parsed
    if meta["result_group"] != "unscored":
        return None
    return scored_path_from_meta(meta, evaluation_dir=evaluation_dir)


def expected_scored_paths(
    model_key: str,
    milestones: list[int],
    n_runs: int,
    judge_repeats: int,
    methods: Iterable[str] | None,
    result_names: Iterable[str] | None,
    run_indices: Iterable[int] | None,
    evaluation_dir: Path | str | None = None,
) -> list[Path] | None:
    if methods is None or result_names is None:
        return None

    milestones = normalize_milestones(milestones)
    runs = list(run_indices) if run_indices is not None else list(range(1, n_runs + 1))

    return [
        build_milestone_result_path(
            model_key=model_key,
            editing_method=method,
            milestones=milestones,
            result_name=result_name,
            run_index=run_index,
            step=step,
            n_runs=n_runs,
            judge_repeats=judge_repeats,
            result_group="scored",
            evaluation_dir=evaluation_dir,
        )
        for method in methods
        for result_name in result_names
        for run_index in runs
        for step in milestones
    ]


def missing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def base_scored_dir(
    model_key: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir is not None else EVALUATION_DIR
    return root / model_key / "base" / "scored" / f"j{judge_repeats}"


def base_scored_path(
    model_key: str,
    result_name: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> Path:
    return base_scored_dir(model_key, judge_repeats, evaluation_dir=evaluation_dir) / f"{result_name}.json"


def base_result_names(
    model_key: str,
    judge_repeats: int,
    evaluation_dir: Path | str | None = None,
) -> list[str]:
    scored_dir = base_scored_dir(model_key, judge_repeats, evaluation_dir=evaluation_dir)
    if not scored_dir.exists():
        return []
    return sorted(path.stem for path in scored_dir.glob("*.json") if path.is_file())


def metric_score(details: list[dict[str, Any]]) -> float:
    return mean(float(record.get("score", 0)) for record in details) if details else 0


def slice_base_metric(
    metric_payload: dict[str, Any] | None,
    selected_indices: list[int],
    step: int,
    dataset_rows: int,
    result_name: str,
    metric_name: str,
) -> dict[str, Any] | None:
    if metric_payload is None:
        return None

    details = list(metric_payload.get("details", []))
    if dataset_rows <= 0:
        raise ValueError(f"Invalid baseline dataset_rows={dataset_rows} for result={result_name}")
    if len(details) % dataset_rows != 0:
        raise ValueError(
            f"Cannot slice baseline result={result_name} metric={metric_name}: "
            f"details={len(details)} is not divisible by dataset_rows={dataset_rows}"
        )

    records_per_row = len(details) // dataset_rows
    sliced_details: list[dict[str, Any]] = []
    for row_index in selected_indices[:step]:
        start = row_index * records_per_row
        sliced_details.extend(details[start:start + records_per_row])

    return {
        "score": metric_score(sliced_details),
        "details": sliced_details,
    }


def slice_base_result(
    result_payload: dict[str, Any],
    selected_indices: list[int],
    step: int,
    dataset_rows: int,
    result_name: str,
) -> dict[str, Any]:
    return {
        metric_name: slice_base_metric(
            result_payload.get(metric_name),
            selected_indices=selected_indices,
            step=step,
            dataset_rows=dataset_rows,
            result_name=result_name,
            metric_name=metric_name,
        )
        for metric_name in ("success", "locality", "portability")
    }
