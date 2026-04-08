from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config


def _resolve_project_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[3]
    return str((project_root / path).resolve())


@dataclass
class MemitConfig(Config):
    # Method
    layers: list[int] = field(default_factory=list)
    layer_selection: str = "all"
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 25
    v_lr: float = 5e-1
    v_loss_layer: int = 27
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4
    kl_factor: float = 0.0625
    mom2_adjustment: bool = True
    mom2_update_weight: float = 15000

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
    stats_dir: str = "data/layer_stats"

    # Defaults
    max_length: int = 40
    batch_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.stats_dir = _resolve_project_path(self.stats_dir)
