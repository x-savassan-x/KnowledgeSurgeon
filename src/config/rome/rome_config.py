from dataclasses import dataclass
from typing import List

from ..config import Config


@dataclass
class RomeConfig(Config):
    # Which layers to apply ROME to (typically a single layer)
    layers: List[int]

    # Strategy for locating the fact token:
    #   'subject_last'  – last token of the subject span
    #   'last'          – last token of the full prompt
    fact_token: str

    # Right-vector (v) optimization
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float

    # Whether to apply C^{-1} correction to the left vector
    mom2_adjustment: bool

    # [(length, n_gen), ...] controls context template generation
    context_template_length_params: List[List[int]]

    # Module templates (dot-notation with {} for layer index)
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Second-moment covariance statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    stats_dir: str

    max_length: int = 40
