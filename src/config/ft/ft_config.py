from dataclasses import dataclass, field
from typing import Literal

from ..config import Config


@dataclass
class FtConfig(Config):
    # Transformer layers whose selected weights are optimized.
    layers: list[int] = field(default_factory=list)

    # Number of optimization steps per edit batch.
    num_steps: int = 25

    # Optimizer learning rate.
    lr: float = 5e-4

    # L2 weight decay applied during optimization.
    weight_decay: float = 0

    # Weight assigned to the KL preservation loss.
    kl_factor: float = 0

    # Optional maximum norm for each learned parameter update.
    norm_constraint: float | None = None

    # Loss variant: target_new is FT-M; prompt_last is FT-L.
    objective_optimization: Literal["target_new", "prompt_last"] = "target_new"

    # Number of training examples processed per optimizer batch.
    batch_size: int = 64

    # Maximum token length used while optimizing edits.
    max_length: int = 40

    # Dot-path template for the weight-bearing module rewritten at each layer.
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.objective_optimization not in {"target_new", "prompt_last"}:
            raise ValueError(
                "objective_optimization must be 'target_new' (FT-M) or "
                "'prompt_last' (FT-L)."
            )
