from dataclasses import dataclass, field
from typing import Literal

from ..config import Config


@dataclass
class GraceConfig(Config):
    # Learning rate used to optimize each stored replacement value.
    edit_lr: float = 1.0

    # Number of optimization steps for each replacement value.
    n_iter: int = 50

    # Initial radius of a new codebook entry.
    eps: float = 1.0

    # Distance metric used to retrieve the nearest codebook key.
    dist_fn: Literal["euc", "mmd", "cos"] = "euc"

    # Initialize replacement values from scratch or from the intercepted state.
    val_init: Literal["cold", "warm"] = "cold"

    # Train replacement values with gradient descent or perturbations.
    val_train: Literal["sgd", "pert"] = "sgd"

    # Optional regularization applied while training replacement values.
    val_reg: str = "None"

    # Optimization stopping strategy for replacement-value training.
    reg: str = "early_stop"

    # Token positions whose hidden states are replaced after retrieval.
    replacement: Literal["replace_last", "replace_all", "replace_prompt"] = "replace_last"

    # Strategy used to update codebook entry radii after collisions.
    eps_expand: str = "coverage"

    # Number of perturbations used by perturbation-based value training.
    num_pert: int = 8

    # Dropout probability applied while optimizing replacement values.
    dropout: float = 0.0

    # Model parameter paths that identify the intercepted module.
    inner_params: list[str] = field(default_factory=list)

    # Number of edit requests processed together.
    batch_size: int = 1

    # Maximum token length used while learning an edit.
    max_length: int = 30
