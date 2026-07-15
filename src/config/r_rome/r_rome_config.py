from dataclasses import dataclass, field

from ..config import Config, resolve_project_path


@dataclass
class RRomeConfig(Config):
    # Method
    layers: list[int] = field(default_factory=list)
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = 47
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 4
    kl_factor: float = 0.0625
    mom2_adjustment: bool = False
    context_template_length_params: list[list[int]] = field(default_factory=lambda: [[5, 10], [10, 10]])
    max_length: int = 40

    # Module templates
    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

    # Statistics
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    stats_dir: str = "data/assets/layer_stats"

    # r-ROME defaults: consistently use random-prefix averaged keys.
    enable_prompt_keys: bool = False
    enable_random_prefix_keys: bool = True
    original_implementation: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.stats_dir = resolve_project_path(self.stats_dir)
