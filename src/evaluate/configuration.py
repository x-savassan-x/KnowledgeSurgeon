from pathlib import Path
from typing import Any

import yaml

from .io import normalize_eval_type


CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation configuration not found: {path}")
    with open(path, encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Evaluation configuration must be a mapping: {path}")
    return payload


def load_judge_configuration() -> dict[str, Any]:
    config = _load_yaml(CONFIG_DIR / "judge.yaml")
    config["eval_type"] = normalize_eval_type(config.get("eval_type", "output"))

    judge = config.get("judge")
    worker = config.get("worker")
    if not isinstance(judge, dict):
        raise TypeError("judge.yaml must define a judge mapping")
    if not isinstance(worker, dict):
        raise TypeError("judge.yaml must define a worker mapping")
    if int(judge.get("repeats", 0)) <= 0:
        raise ValueError("judge.repeats must be positive")
    return config


def load_experiment_configuration(eval_type: str) -> dict[str, Any]:
    eval_type = normalize_eval_type(eval_type)
    shared = _load_yaml(CONFIG_DIR / "shared.yaml")
    experiment = _load_yaml(CONFIG_DIR / f"{eval_type}.yaml")
    overlap = shared.keys() & experiment.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Shared and experiment configuration keys overlap: {names}")

    judge = load_judge_configuration()["judge"]
    return {
        **shared,
        **experiment,
        "eval_type": eval_type,
        "judge_repeats": int(judge["repeats"]),
    }


def load_selected_evaluation_configuration() -> tuple[dict[str, Any], dict[str, Any]]:
    judge_config = load_judge_configuration()
    experiment_config = load_experiment_configuration(judge_config["eval_type"])
    return experiment_config, judge_config
