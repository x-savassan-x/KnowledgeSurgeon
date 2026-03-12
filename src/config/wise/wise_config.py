from dataclasses import dataclass, field
from typing import Optional, Union
from ..config import Config


@dataclass
class WiseConfig(Config):
    edit_lr: float = 1.0
    n_iter: int = 40
    objective_optimization: str = "only_label"
    mask_ratio: float = 0.2
    alpha: float = 5.0
    beta: float = 20.0
    gamma: float = 10.0
    act_ratio: float = 0.8
    merge_freq: int = 1000
    retrieve: bool = True
    replay: bool = False
    save_freq: Union[int, None] = 500
    merge_alg: str = "ties"
    norm_constraint: float = 1.0
    inner_params: list[str] = field(default_factory=list)
    weights: Union[float, None] = 1.0
    densities: Union[float, None] = 0.53
    batch_size: int = 1
    max_length: int = 30
    sequential_edit: bool = True
