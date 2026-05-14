from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..memit import MemitConfig
from ..rome import RomeConfig


def _resolve_project_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[3]
    return str((project_root / path).resolve())


@dataclass
class MnemoConfig(Config):
    online_config_path: str = ""
    batch_config_path: str = ""
    buffer_capacity: int = 100
    auto_consolidate: bool = True

    online_config: RomeConfig = field(init=False)
    batch_config: MemitConfig = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.buffer_capacity <= 0:
            raise ValueError("Mnemo buffer_capacity must be positive.")

        self.online_config_path = _resolve_project_path(self.online_config_path)
        self.batch_config_path = _resolve_project_path(self.batch_config_path)

        from .. import load_config

        online_config = load_config(self.online_config_path)
        batch_config = load_config(self.batch_config_path)

        if not isinstance(online_config, RomeConfig):
            raise TypeError("Mnemo online_config_path must point to a ROME config.")
        if not isinstance(batch_config, MemitConfig):
            raise TypeError("Mnemo batch_config_path must point to a MEMIT config.")

        self.online_config = self._synchronize_child_config(online_config)
        self.batch_config = self._synchronize_child_config(batch_config)

    def _synchronize_child_config(self, child_config: Config):
        child_config.model_name = self.model_name
        child_config.gpu_id = self.gpu_id
        child_config.device = self.device
        child_config.fp16 = self.fp16
        child_config.model_parallel = self.model_parallel
        child_config.device_map = self.device_map
        return child_config
