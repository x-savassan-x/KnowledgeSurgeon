from dataclasses import dataclass, field
from typing import Optional
from ..config import Config


@dataclass
class FtConfig(Config):
    # Method
    layers: list[int] = field(default_factory=list)
    num_steps: int = 25
    lr: float = 5e-4
    weight_decay: float = 0
    kl_factor: float = 0
    norm_constraint: Optional[float] = None
    # We provide two implementations (objective_optimization):
    #   1. prompt_last: the method of ROME's original paper,
    #      which calculates NLL loss through the last token of the input.
    #   2. target_new: the standard autoregressive method,
    #      using the cross-entropy loss function.
    # In our survey paper (https://arxiv.org/abs/2401.01286):
    # "prompt_last" corresponds to FT-L, and "target_new" to FT-M.
    objective_optimization: str = "prompt_last"

    # Defaults
    batch_size: int = 64
    max_length: int = 40

    # Module templates
    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""
