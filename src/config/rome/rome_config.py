from dataclasses import dataclass, field

from ..config import Config, resolve_project_path


@dataclass
class RomeConfig(Config):
    # Transformer layers whose MLP weights receive rank-one updates.
    layers: list[int] = field(default_factory=list)

    # Subject token position used to extract the rewrite key.
    fact_token: str = "subject_last"

    # Maximum optimization steps used to solve the target value vector.
    v_num_grad_steps: int = 20

    # Learning rate for target value optimization.
    v_lr: float = 5e-1

    # Layer where the target prediction loss is measured.
    v_loss_layer: int = 47

    # Weight decay applied to the optimized target value.
    v_weight_decay: float = 0.5

    # Maximum target-value norm relative to the original activation norm.
    clamp_norm_factor: float = 4

    # Weight assigned to the KL preservation loss.
    kl_factor: float = 0.0625

    # Apply the inverse second moment to the rewrite key.
    mom2_adjustment: bool = False

    # Pairs of context length and sample count used to build key templates.
    context_template_length_params: list[list[int]] = field(default_factory=lambda: [[5, 10], [10, 10]])

    # Maximum token length used during value optimization.
    max_length: int = 40

    # Dot-path template for the MLP weight rewritten at each layer.
    rewrite_module_tmp: str = ""

    # Dot-path template for a transformer layer.
    layer_module_tmp: str = ""

    # Dot-path template for a layer's MLP module.
    mlp_module_tmp: str = ""

    # Dot-path template for a layer's attention module.
    attn_module_tmp: str = ""

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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stats_dir = resolve_project_path(self.stats_dir)
