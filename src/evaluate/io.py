import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def _normalize_milestones(milestones: list[int]) -> list[int]:
    if not milestones:
        raise ValueError("milestones must not be empty")

    normalized = sorted({int(step) for step in milestones})
    if normalized[0] <= 0:
        raise ValueError("milestones must be positive integers")

    return normalized


def milestones_to_str(milestones: list[int]) -> str:
    return "-".join(str(step) for step in _normalize_milestones(milestones))


def build_result_path(
    dataset_variant: str,
    edited_model: str,
    editing_method: str,
    milestones: list[int],
    dataset_name: str,
    evaluation_dir: Path | str | None = None,
) -> Path:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    milestones_dir = milestones_to_str(milestones)

    return root / dataset_variant / edited_model / editing_method / milestones_dir / f"{dataset_name}.json"


def save_benchmark_results(
    results: list[dict[str, Any]],
    dataset_variant: str,
    edited_model: str,
    editing_method: str,
    milestones: list[int],
    dataset_name: str,
    evaluation_dir: Path | str | None = None,
) -> Path:
    path = build_result_path(
        dataset_variant=dataset_variant,
        edited_model=edited_model,
        editing_method=editing_method,
        milestones=milestones,
        dataset_name=dataset_name,
        evaluation_dir=evaluation_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        # Keep payload exactly as produced by the benchmark pipeline.
        json.dump(results, file, indent=2, ensure_ascii=False)

    return path


def iter_result_files(evaluation_dir: Path | str | None = None) -> list[Path]:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_result_file(
    path: Path,
    evaluation_dir: Path | str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    root = Path(evaluation_dir) if evaluation_dir else EVALUATION_DIR

    try:
        relative = path.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) != 5:
        return None

    dataset_variant, edited_model, editing_method, milestones_str, filename = parts
    if not filename.endswith(".json"):
        return None

    dataset_name = filename[:-5]
    if not dataset_name:
        return None

    try:
        milestones = [int(step) for step in milestones_str.split("-")]
    except ValueError:
        return None

    milestones = _normalize_milestones(milestones)

    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        return None

    meta = {
        "variant": dataset_variant,
        "model": edited_model,
        "method": editing_method,
        "milestones": milestones,
        "dataset_name": dataset_name,
    }
    return meta, payload
