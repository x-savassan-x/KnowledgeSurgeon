from dataclasses import dataclass, field
from ..config import Config, resolve_project_path


@dataclass
class WiseConfig(Config):
    # Tracked irrelevant-query bank used by the WISE objective (Equation 5).
    loc_data_path: str = "src/editors/wise/loc_data.json"

    # Learning rate used to optimize the side-memory weights.
    edit_lr: float = 1.0

    # Number of optimization steps per edit batch.
    n_iter: int = 40

    # Token subset included in the edit objective.
    objective_optimization: str = "only_label"

    # Fraction of activations masked when constructing edit features.
    mask_ratio: float = 0.2

    # Maximum allowed activation drift for out-of-scope examples.
    alpha: float = 5.0

    # Minimum desired activation change for in-scope examples.
    beta: float = 20.0

    # Minimum separation between in-scope and out-of-scope distances.
    gamma: float = 10.0

    # Optional compact override for alpha, beta, and gamma.
    act_margin: list[float] | None = None

    # Fraction of the activation margin applied during routing.
    act_ratio: float = 0.8

    # Number of edits between side-memory merges.
    merge_freq: int = 1000

    # Enable retrieval from stored side-memory experts.
    retrieve: bool = True

    # Replay earlier edit examples during optimization.
    replay: bool = False

    # Optional interval for saving intermediate editor state.
    save_freq: int | None = 500

    # Algorithm used to merge side-memory weights into the main model.
    merge_alg: str = "ties"

    # Maximum norm allowed for a learned update.
    norm_constraint: float = 1.0

    # Model parameter paths edited by WISE.
    inner_params: list[str] = field(default_factory=list)

    # Optional weight assigned to each model during merging.
    weights: float | None = 1.0

    # Optional parameter density retained during merging.
    densities: float | None = 0.53

    # Number of edit requests processed together.
    batch_size: int = 1

    # Maximum token length used while learning an edit.
    max_length: int = 30

    # Apply edits sequentially instead of as independent model copies.
    sequential_edit: bool = True

    # Format edit prompts through the tokenizer's chat template.
    use_chat_template: bool = False

    def __post_init__(self) -> None:
        self.loc_data_path = resolve_project_path(self.loc_data_path)
        if self.act_margin is not None:
            if len(self.act_margin) != 3:
                raise ValueError("WISE act_margin must contain [alpha, beta, gamma].")
            self.alpha, self.beta, self.gamma = self.act_margin
        super().__post_init__()
