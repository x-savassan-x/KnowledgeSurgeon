from dataclasses import dataclass, field

from ..config import Config, resolve_project_path


@dataclass
class AlphaEditConfig(Config):
    # Transformer layers whose MLP weights receive projected updates.
    layers: list[int] = field(default_factory=list)

    # Subject token position used to extract the rewrite key.
    fact_token: str = "subject_last"

    # Maximum optimization steps used to solve each target value vector.
    v_num_grad_steps: int = 20

    # Learning rate for target value optimization.
    v_lr: float = 5e-1

    # Layer where the target prediction loss is measured.
    v_loss_layer: int = 47

    # Weight decay applied to optimized target values.
    v_weight_decay: float = 0.5

    # Maximum target-value norm relative to the original activation norm.
    clamp_norm_factor: float = 0.75

    # Weight assigned to the KL preservation loss.
    kl_factor: float = 0.0625

    # Dot-path template for the MLP weight rewritten at each layer.
    rewrite_module_tmp: str = ""

    # Dot-path template for a transformer layer.
    layer_module_tmp: str = ""

    # Dot path for the model's final normalization module.
    ln_f_module: str = ""

    # Dot path for the model's language-model head.
    lm_head_module: str = ""

    # Dataset used to estimate the activation second moment.
    mom2_dataset: str = "wikipedia"

    # Number of samples used to estimate the activation second moment.
    mom2_n_samples: int = 100000

    # Storage dtype for activation statistics.
    mom2_dtype: str = "float32"

    # Directory containing cached activation statistics.
    stats_dir: str = "data/assets/layer_stats"

    # Eigenvalue cutoff used to construct the protected null space.
    nullspace_threshold: float = 2e-2

    # L2 regularization coefficient used when solving weight updates.
    L2: float = 10

    # Path to the cached AlphaEdit null-space projector.
    P_loc: str = "data/assets/alpha_edit_projectors/null_space_project.pt"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stats_dir = resolve_project_path(self.stats_dir)
        self.P_loc = resolve_project_path(self.P_loc)
