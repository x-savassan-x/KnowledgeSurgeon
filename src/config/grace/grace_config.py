from dataclasses import dataclass, field
from typing import Literal

from ..config import Config


@dataclass
class GraceConfig(Config):
    # Experiments
    edit_lr: float = 1.0
    n_iter: int = 50

    # Method
    eps: float = 1.0
    dist_fn: Literal["euc", "mmd", "cos"] = "euc"  # euc, mmd, cos
    val_init: Literal["cold", "warm"] = "cold"  # cold, warm
    val_train: Literal["sgd", "pert"] = "sgd"  # sgd, pert
    val_reg: str = "None"  # early
    reg: str = "early_stop"  # early_stop
    replacement: Literal["replace_last", "replace_all", "replace_prompt"] = "replace_last"  # replace_last, replace_all, replace_prompt
    eps_expand: str = "coverage"  # coverage, moving_avg, decay
    num_pert: int = 8  # only matters when using perturbation training
    dropout: float = 0.0

    # Module templates
    inner_params: list[str] = field(default_factory=list)

    # Defaults
    batch_size: int = 1
    max_length: int = 30
