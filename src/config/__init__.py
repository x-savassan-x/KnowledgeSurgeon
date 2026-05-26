import yaml

from .config import Config

from .ft import FtConfig
from .rome import RomeConfig
from .memit import MemitConfig
from .grace import GraceConfig
from .wise import WiseConfig
from .alpha_edit import AlphaEditConfig
# from .mnemo import MnemoConfig


CONFIG_REGISTRY: dict[str, type[Config]] = {
    "FT": FtConfig,
    "ROME": RomeConfig,
    "MEMIT": MemitConfig,
    "GRACE": GraceConfig,
    "WISE": WiseConfig,
    "AlphaEdit": AlphaEditConfig,
    # "Mnemo": MnemoConfig,
}


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alg_name = config["alg_name"]
    if alg_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unregistered config: {alg_name}.")

    config_cls = CONFIG_REGISTRY[alg_name]
    return config_cls(**config)
