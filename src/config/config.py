from dataclasses import dataclass, field
from pathlib import Path
import torch


def resolve_project_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / path).resolve())


"""
Several params take dot-notation paths into the model's layer hierarchy

To get all paths info for a loaded model:
```
rows = []
for path, module in model.named_modules():
    shapes = {n: tuple(p.shape) for n, p in module.named_parameters(recurse=False)}
    rows.append((f"{path}.{type(module).__name__}", str(shapes)))

col1_w = max(len(r[0]) for r in rows)
for name, shapes in rows:
    print(f"{name:<{col1_w}}  {shapes}")
```
"""
@dataclass
class Config:
    # Name of the editing algorithm
    alg_name: str

    # HuggingFace model ID or local path to the model directory
    model_name: str

    # Stable project slug used for model-specific assets and evaluation paths.
    model_key: str

    # CUDA device index to load the model on
    gpu_id: int

    # Materialized device used everywhere in code after config loading.
    # For current single-GPU workflow, this is always cuda:{gpu_id}.
    device: torch.device = field(init=False)

    # When True, load model in bfloat16 instead of float32
    fp16: bool

    # When True, load large models sharded across visible GPUs via HF device_map.
    model_parallel: bool = False

    # Model-specific limits for activation statistics used by ROME-family methods.
    layer_stats_max_length: int | None = None

    def __post_init__(self):
        if not self.model_key or Path(self.model_key).name != self.model_key:
            raise ValueError("model_key must be a non-empty path-safe slug.")
        if self.layer_stats_max_length is not None and self.layer_stats_max_length <= 0:
            raise ValueError("layer_stats_max_length must be positive when set.")
        self.device = torch.device(f"cuda:{self.gpu_id}")
