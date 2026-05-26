from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config


def _resolve_project_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[3]
    return str((project_root / path).resolve())


@dataclass
class MnemoOnlineConfig:
    alg_name: str = "ROME"

    layers: list[int] = field(default_factory=list)
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = 47
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 4
    kl_factor: float = 0.0625
    mom2_adjustment: bool = False
    context_template_length_params: list[list[int]] = field(
        default_factory=lambda: [[5, 10], [10, 10]]
    )
    max_length: int = 40

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

    def __post_init__(self):
        self.stats_dir = _resolve_project_path(self.stats_dir)


@dataclass
class MnemoBatchConfig:
    alg_name: str = "MEMIT"

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

    rewrite_module_tmp: str = ""
    layer_module_tmp: str = ""
    mlp_module_tmp: str = ""
    attn_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    stats_dir: str = "data/layer_stats"

    max_length: int = 40
    batch_size: int = 1

    def __post_init__(self):
        self.stats_dir = _resolve_project_path(self.stats_dir)


@dataclass(kw_only=True)
class MnemoConfig(Config):
    online: dict[str, Any] | MnemoOnlineConfig
    batch: dict[str, Any] | MnemoBatchConfig
    buffer_capacity: int = 100
    auto_consolidate: bool = True

    online_config: MnemoOnlineConfig = field(init=False)
    batch_config: MnemoBatchConfig = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.buffer_capacity <= 0:
            raise ValueError("Mnemo buffer_capacity must be positive.")

        self.online_config = self._coerce_online_config(self.online)
        self.batch_config = self._coerce_batch_config(self.batch)

        self._validate_phase_algorithms()

    @staticmethod
    def _coerce_online_config(
        phase_config: dict[str, Any] | MnemoOnlineConfig,
    ) -> MnemoOnlineConfig:
        if isinstance(phase_config, MnemoOnlineConfig):
            return phase_config
        if not isinstance(phase_config, dict):
            raise TypeError("Mnemo online configuration must be a mapping.")
        return MnemoOnlineConfig(**phase_config)

    @staticmethod
    def _coerce_batch_config(
        phase_config: dict[str, Any] | MnemoBatchConfig,
    ) -> MnemoBatchConfig:
        if isinstance(phase_config, MnemoBatchConfig):
            return phase_config
        if not isinstance(phase_config, dict):
            raise TypeError("Mnemo batch configuration must be a mapping.")
        return MnemoBatchConfig(**phase_config)

    def _validate_phase_algorithms(self):
        if self.online_config.alg_name != "ROME":
            raise ValueError(
                f"Unsupported Mnemo online phase algorithm: {self.online_config.alg_name}."
            )
        if self.batch_config.alg_name != "MEMIT":
            raise ValueError(
                f"Unsupported Mnemo batch phase algorithm: {self.batch_config.alg_name}."
            )
