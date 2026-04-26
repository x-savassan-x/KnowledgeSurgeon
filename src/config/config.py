from dataclasses import dataclass, field
from typing import Any

import torch


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

    # CUDA device index to load the model on
    gpu_id: int

    # Materialized device used everywhere in code after config loading.
    # For current single-GPU workflow, this is always cuda:{gpu_id}.
    device: torch.device = field(init=False)

    # When True, load model in bfloat16 instead of float32
    fp16: bool

    # When True, load large models sharded across visible GPUs via HF device_map.
    model_parallel: bool = False

    # Optional explicit HF device_map. If omitted with model_parallel=True, use "auto".
    device_map: Any | None = None

    def __post_init__(self):
        self.device = torch.device(f"cuda:{self.gpu_id}")
