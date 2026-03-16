import yaml

from .config import Config

from .ft import FtConfig
from .rome import RomeConfig
from .grace import GraceConfig
from .wise import WiseConfig
from .alpha_edit import AlphaEditConfig


CONFIG_REGISTRY: dict[str, type[Config]] = {
    "FT": FtConfig,
    "ROME": RomeConfig,
    "GRACE": GraceConfig,
    "WISE": WiseConfig,
    "AlphaEdit": AlphaEditConfig
}


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alg_name = config["alg_name"]
    if alg_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unregistered config: {alg_name}.")

    config_cls = CONFIG_REGISTRY[alg_name]
    return config_cls(**config)
