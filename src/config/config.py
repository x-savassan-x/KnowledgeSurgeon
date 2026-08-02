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
    # Name used to select the editing algorithm.
    alg_name: str

    # Hugging Face model ID or local model directory.
    model_name: str

    # Stable slug used for model-specific assets and evaluation paths.
    model_key: str

    # CUDA device index used by the editor and model.
    gpu_id: int

    # Runtime device derived from gpu_id after configuration loading.
    device: torch.device = field(init=False)

    # Load model weights in bfloat16 instead of float32.
    fp16: bool

    # Run a disposable gradient pass to compile model kernels before timing edits.
    warmup: bool = False

    # Shard the model across visible GPUs through the Hugging Face device map.
    model_parallel: bool = False

    # Optional token limit for activation statistics used by ROME-family methods.
    layer_stats_max_length: int | None = None

    def __post_init__(self) -> None:
        if not self.model_key or Path(self.model_key).name != self.model_key:
            raise ValueError("model_key must be a non-empty path-safe slug.")
        if self.layer_stats_max_length is not None and self.layer_stats_max_length <= 0:
            raise ValueError("layer_stats_max_length must be positive when set.")
        self.device = torch.device(f"cuda:{self.gpu_id}")
