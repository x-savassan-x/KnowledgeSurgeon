from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .ft import FtConfig
from .rome import RomeConfig
from .memit import MemitConfig
from .grace import GraceConfig
from .wise import WiseConfig
from .alpha_edit import AlphaEditConfig
from .mnemo import MnemoConfig


CONFIG_REGISTRY: dict[str, type[Config]] = {
    "FT": FtConfig,
    "ROME": RomeConfig,
    "MEMIT": MemitConfig,
    "GRACE": GraceConfig,
    "WISE": WiseConfig,
    "AlphaEdit": AlphaEditConfig,
    "Mnemo": MnemoConfig,
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return data


def _resolve_config_path(path: str, parent: Path) -> Path:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = parent / config_path
    return config_path.resolve()


def _merge_model_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    model_config = config.pop("model_config", None)
    if model_config is None:
        return config

    if isinstance(model_config, str):
        base_config = _read_yaml(_resolve_config_path(model_config, config_path.parent))
    elif isinstance(model_config, dict):
        base_config = dict(model_config)
    else:
        raise TypeError("model_config must be a path string or mapping.")

    return {**base_config, **config}


def load_config(config_path) -> Config:
    config_path = Path(config_path).resolve()
    config = _merge_model_config(_read_yaml(config_path), config_path)

    alg_name = config["alg_name"]
    if alg_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unregistered config: {alg_name}.")

    config_cls = CONFIG_REGISTRY[alg_name]
    return config_cls(**config)
