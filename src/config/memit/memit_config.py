from dataclasses import dataclass, field
from ..config import Config


@dataclass
class MemitConfig(Config):
    layers: list[int] = field(default_factory=list)
    layer_selection: str = "all"
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = 47
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625
    mom2_adjustment: bool = True
    mom2_update_weight: float = 20000
    context_template_length_params: list[list[int]] = field(default_factory=lambda: [[5, 10], [10, 10]])
    max_length: int = 40
    batch_size: int = 1

    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    stats_dir: str = "./data/stats"
