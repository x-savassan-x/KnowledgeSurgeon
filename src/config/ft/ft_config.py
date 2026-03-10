from dataclasses import dataclass
from typing import List

from ..config import Config


@dataclass
class FtConfig(Config):
    # Which layers to fine-tune
    layers: List[int]

    # Training loop
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float  # clamp weight change to ±eps of original

    # Loss mode:
    #   'prompt_last'  – predict target from last input token (FT-L)
    #   'target_new'   – standard autoregressive CE on target tokens (FT-M)
    objective_optimization: str

    # Module templates (dot-notation with {} for layer index)
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    batch_size: int = 64
    max_length: int = 40
