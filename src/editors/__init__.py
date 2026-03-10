import yaml

from .editor import Editor
from .ft import FT
from .rome import ROME
from .memit import MEMIT
from .grace import GRACE
from .wise import WISE
from .alpha_edit import AlphaEdit

from ..config import (
    FtConfig,
    RomeConfig,
    MemitConfig,
    GraceConfig,
    WiseConfig,
    AlphaEditConfig,
)


EDITOR_REGISTRY = {
    "FT": (FT, FtConfig),
    "ROME": (ROME, RomeConfig),
    "MEMIT": (MEMIT, MemitConfig),
    "GRACE": (GRACE, GraceConfig),
    "WISE": (WISE, WiseConfig),
    "AlphaEdit": (AlphaEdit, AlphaEditConfig),
}


def load_editor(config_path: str) -> Editor:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alg_name = config["alg_name"]
    if alg_name not in EDITOR_REGISTRY:
        raise ValueError(f"Unknown editing method: {alg_name}.")

    editor_cls, config_cls = EDITOR_REGISTRY[alg_name]
    return editor_cls(config_cls(**config))
