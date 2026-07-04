import torch

from ..config import load_config
from .editor import Editor
from .editor_utils import EditRequest
from .ft import FT
from .rome import ROME
from .r_rome import RROME
from .memit import MEMIT
from .grace import GRACE
from .wise import WISE
from .alpha_edit import AlphaEdit
from .mnemo import Mnemo


EDITOR_REGISTRY: dict[str, type[Editor]] = {
    "FT": FT,
    "ROME": ROME,
    "R-ROME": RROME,
    "MEMIT": MEMIT,
    "GRACE": GRACE,
    "WISE": WISE,
    "AlphaEdit": AlphaEdit,
    "Mnemo": Mnemo,
}


def load_editor(config_path: str, model=None, tokenizer=None, gpu_id: int | None = None) -> Editor:
    config = load_config(config_path)
    if gpu_id is not None:
        config.gpu_id = int(gpu_id)
        config.device = torch.device(f"cuda:{config.gpu_id}")
        if config.model_parallel:
            config.device_map = {"": config.gpu_id} if config.device_map is None else config.device_map

    alg_name = config.alg_name
    if alg_name not in EDITOR_REGISTRY:
        raise ValueError(f"Unregistered editor: {alg_name}.")

    editor_cls = EDITOR_REGISTRY[alg_name]
    return editor_cls(config, model=model, tokenizer=tokenizer)
