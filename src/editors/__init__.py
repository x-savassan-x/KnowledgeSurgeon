from .editor import Editor
from .edtior_utils import EditRequest

from .ft import FT
from .rome import ROME
from .memit import MEMIT
from .grace import GRACE
from .wise import WISE
from .alpha_edit import AlphaEdit

from ..config import load_config


EDITOR_REGISTRY: dict[str, type[Editor]] = {
    "FT": FT,
    "ROME": ROME,
    "MEMIT": MEMIT,
    "GRACE": GRACE,
    "WISE": WISE,
    "AlphaEdit": AlphaEdit,
}


def load_editor(config_path: str) -> Editor:
    config = load_config(config_path)

    alg_name = config.alg_name
    if alg_name not in EDITOR_REGISTRY:
        raise ValueError(f"Unregistered editor: {alg_name}.")

    editor_cls = EDITOR_REGISTRY[alg_name]
    return editor_cls(config)
