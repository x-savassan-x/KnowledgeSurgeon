from dataclasses import dataclass, field
from ..config import Config


@dataclass
class FtConfig(Config):
    # Method
    layers: list[int] = field(default_factory=list)
    num_steps: int = 25
    lr: float = 5e-4
    weight_decay: float = 0
    kl_factor: float = 0
    norm_constraint: float | None = None

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
