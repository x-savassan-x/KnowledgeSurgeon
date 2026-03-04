import yaml
from .editor import Editor
from .grace import GRACE

from ..hparams import GraceHyperParams


EDITOR_REGISTRY = {
    "GRACE": (GRACE, GraceHyperParams)
}


def load_editor(config_path: str) -> Editor:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    alg_name = config["alg_name"]
    if alg_name not in EDITOR_REGISTRY:
        raise ValueError(f"Unknown editing method: {alg_name}.")

    editor_cls, hparams_cls = EDITOR_REGISTRY[alg_name]
    return editor_cls(hparams_cls(**config))
