from dataclasses import dataclass, field

from ..config import Config, resolve_project_path


@dataclass
class AlphaEditConfig(Config):
    # Method
    layers: list[int] = field(default_factory=list)
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = 47
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625
    # Module templates
    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

    # Statistics
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    stats_dir: str = "data/assets/layer_stats"

    nullspace_threshold: float = 2e-2
    L2: float = 10
    P_loc: str = "data/assets/alpha_edit_projectors/null_space_project.pt"

    def __post_init__(self):
        super().__post_init__()
        self.stats_dir = resolve_project_path(self.stats_dir)
        self.P_loc = resolve_project_path(self.P_loc)
