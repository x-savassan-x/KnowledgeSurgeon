from dataclasses import dataclass, field
from typing import Any

from ..alpha_edit import AlphaEditConfig


@dataclass
class MnemoRouterThresholdConfig:
    score_threshold: float
    margin_threshold: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError("Mnemo router score_threshold must be in [-1, 1].")
        if not 0.0 <= self.margin_threshold <= 2.0:
            raise ValueError("Mnemo router margin_threshold must be in [0, 2].")


def _default_hybrid_thresholds() -> dict[str, float]:
    return {
        "score_threshold": 0.5810490846633911,
        "margin_threshold": 0.06471586972475052,
    }


def _default_external_only_thresholds() -> dict[str, float]:
    return {
        "score_threshold": 0.6538538336753845,
        "margin_threshold": 0.09644604474306107,
    }


@dataclass
class MnemoRouterConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    fp16: bool = True
    eager_load: bool = True
    max_length: int = 384
    batch_size: int = 32
    gate: str = "score_margin"
    hybrid: dict[str, float] | MnemoRouterThresholdConfig = field(
        default_factory=_default_hybrid_thresholds
    )
    external_only: dict[str, float] | MnemoRouterThresholdConfig = field(
        default_factory=_default_external_only_thresholds
    )

    hybrid_config: MnemoRouterThresholdConfig = field(init=False)
    external_only_config: MnemoRouterThresholdConfig = field(init=False)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("Mnemo router model_name must not be empty.")
        if not self.device:
            raise ValueError("Mnemo router device must not be empty.")
        if not isinstance(self.fp16, bool):
            raise TypeError("Mnemo router fp16 must be a boolean.")
        if not isinstance(self.eager_load, bool):
            raise TypeError("Mnemo router eager_load must be a boolean.")
        if self.max_length <= 0:
            raise ValueError("Mnemo router max_length must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Mnemo router batch_size must be positive.")
        if self.gate != "score_margin":
            raise ValueError("Mnemo currently supports only the score_margin router gate.")
        self.hybrid_config = self._coerce_thresholds(self.hybrid)
        self.external_only_config = self._coerce_thresholds(self.external_only)

    @staticmethod
    def _coerce_thresholds(
        config: dict[str, float] | MnemoRouterThresholdConfig,
    ) -> MnemoRouterThresholdConfig:
        if isinstance(config, MnemoRouterThresholdConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo router thresholds must be a mapping.")
        return MnemoRouterThresholdConfig(**config)

    def thresholds_for(self, external_only: bool) -> tuple[float, float]:
        config = self.external_only_config if external_only else self.hybrid_config
        return config.score_threshold, config.margin_threshold


@dataclass
class MnemoDeltaConfig:
    num_steps: int = 50
    learning_rate: float = 1.0
    weight_decay: float = 0.0
    max_norm: float = 40.0
    early_stop_loss: float = 1.0e-3
    target_leading_space: bool = True

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("Mnemo delta num_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Mnemo delta learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("Mnemo delta weight_decay must be nonnegative.")
        if self.max_norm <= 0:
            raise ValueError("Mnemo delta max_norm must be positive.")
        if self.early_stop_loss < 0:
            raise ValueError("Mnemo delta early_stop_loss must be nonnegative.")


def _default_canary_prompts() -> list[str]:
    return [
        "The sky above the quiet field was",
        "After reviewing the evidence, the committee",
        "To prepare a simple meal, first",
        "In a well-designed experiment, researchers",
        "During the winter, the river",
        "The purpose of a public library is",
        "When the meeting ended, everyone",
        "A careful explanation should",
        "On a clear morning, the city",
        "The small wooden box contained",
        "If the temperature continues to rise, the ice",
        "Before making a decision, it is useful to",
        "Most people learn a new skill by",
        "The report concluded that the results",
        "At the end of the journey, they",
        "Because the instructions were incomplete, the team",
    ]


@dataclass
class MnemoSafetyConfig:
    enabled: bool = True
    start_after_edits: int = 500
    validation_microbatch_size: int = 4
    canary_prompts: list[str] = field(default_factory=_default_canary_prompts)
    canary_sample_size: int = 16
    retention_reservoir_size: int = 256
    retention_sample_size: int = 64
    sphere_sample_size: int = 64
    sphere_chunk_size: int = 16
    sphere_seed: int = 0
    sphere_exponent: float = 1.0
    tensor_chunk_elements: int = 1_000_000
    warning_votes_required: int = 2

    relative_update_norm_warning: float = 0.10
    relative_update_norm_hard_limit: float = 1.0

    step_kl_warning: float = 0.05
    step_kl_hard_limit: float = 0.20
    cumulative_kl_warning: float = 0.10
    cumulative_kl_hard_limit: float = 0.50
    step_nll_warning: float = 0.25
    step_nll_hard_limit: float = 0.75
    cumulative_nll_warning: float = 0.50
    cumulative_nll_hard_limit: float = 1.50

    current_batch_min_success: float = 0.80
    retention_min_success: float = 0.80
    retention_max_mean_drop: float = 0.20

    sphere_step_warning: float = 0.10
    sphere_step_hard_limit: float = 0.25
    sphere_cumulative_warning: float = 0.20
    sphere_cumulative_hard_limit: float = 0.50

    def __post_init__(self) -> None:
        if not self.canary_prompts:
            raise ValueError("Mnemo safety canary_prompts must not be empty.")
        normalized_canaries = [prompt.strip() for prompt in self.canary_prompts]
        if any(not prompt for prompt in normalized_canaries):
            raise ValueError("Mnemo safety canary prompts must not be empty.")
        if len(set(normalized_canaries)) != len(normalized_canaries):
            raise ValueError("Mnemo safety canary prompts must be unique.")
        self.canary_prompts = normalized_canaries
        positive_integer_fields = (
            "validation_microbatch_size",
            "canary_sample_size",
            "retention_reservoir_size",
            "retention_sample_size",
            "sphere_sample_size",
            "sphere_chunk_size",
            "tensor_chunk_elements",
            "warning_votes_required",
        )
        for name in positive_integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"Mnemo safety {name} must be positive.")
        if self.start_after_edits < 0:
            raise ValueError("Mnemo safety start_after_edits must be nonnegative.")
        self.canary_sample_size = min(
            self.canary_sample_size,
            len(self.canary_prompts),
        )
        if self.retention_sample_size > self.retention_reservoir_size:
            raise ValueError(
                "Mnemo retention_sample_size cannot exceed retention_reservoir_size."
            )
        if self.sphere_exponent <= 0:
            raise ValueError("Mnemo sphere_exponent must be positive.")

        probability_fields = (
            "current_batch_min_success",
            "retention_min_success",
        )
        for name in probability_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Mnemo safety {name} must be in [0, 1].")

        nonnegative_fields = (
            "relative_update_norm_warning",
            "relative_update_norm_hard_limit",
            "step_kl_warning",
            "step_kl_hard_limit",
            "cumulative_kl_warning",
            "cumulative_kl_hard_limit",
            "step_nll_warning",
            "step_nll_hard_limit",
            "cumulative_nll_warning",
            "cumulative_nll_hard_limit",
            "retention_max_mean_drop",
            "sphere_step_warning",
            "sphere_step_hard_limit",
            "sphere_cumulative_warning",
            "sphere_cumulative_hard_limit",
        )
        for name in nonnegative_fields:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"Mnemo safety {name} must be nonnegative.")

        limit_pairs = (
            ("relative_update_norm_warning", "relative_update_norm_hard_limit"),
            ("step_kl_warning", "step_kl_hard_limit"),
            ("cumulative_kl_warning", "cumulative_kl_hard_limit"),
            ("step_nll_warning", "step_nll_hard_limit"),
            ("cumulative_nll_warning", "cumulative_nll_hard_limit"),
            ("sphere_step_warning", "sphere_step_hard_limit"),
            ("sphere_cumulative_warning", "sphere_cumulative_hard_limit"),
        )
        for warning_name, hard_name in limit_pairs:
            if float(getattr(self, warning_name)) > float(getattr(self, hard_name)):
                raise ValueError(
                    f"Mnemo safety {warning_name} cannot exceed {hard_name}."
                )


@dataclass
class MnemoConfig(AlphaEditConfig):
    consolidation_cycle: int = 100
    external_only: bool = False
    diagnostics: bool = False
    intervention_block: int = 18
    intervention_module_tmp: str = "model.layers.{}"
    intervention_strength: float = 1.0
    router: dict[str, Any] | MnemoRouterConfig = field(default_factory=dict)
    delta: dict[str, Any] | MnemoDeltaConfig = field(default_factory=dict)
    safety: dict[str, Any] | MnemoSafetyConfig = field(default_factory=dict)

    router_config: MnemoRouterConfig = field(init=False)
    delta_config: MnemoDeltaConfig = field(init=False)
    safety_config: MnemoSafetyConfig = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.consolidation_cycle <= 0:
            raise ValueError("Mnemo consolidation_cycle must be positive.")
        if not isinstance(self.external_only, bool):
            raise TypeError("Mnemo external_only must be a boolean.")
        if not isinstance(self.diagnostics, bool):
            raise TypeError("Mnemo diagnostics must be a boolean.")
        if self.intervention_block < 0:
            raise ValueError("Mnemo intervention_block must be nonnegative.")
        if "{}" not in self.intervention_module_tmp:
            raise ValueError("Mnemo intervention_module_tmp must contain '{}'.")
        if not 0.0 < self.intervention_strength <= 1.0:
            raise ValueError("Mnemo intervention_strength must be in (0, 1].")
        self.router_config = self._coerce_router_config(self.router)
        self.delta_config = self._coerce_delta_config(self.delta)
        self.safety_config = self._coerce_safety_config(self.safety)

    @staticmethod
    def _coerce_router_config(
        config: dict[str, Any] | MnemoRouterConfig,
    ) -> MnemoRouterConfig:
        if isinstance(config, MnemoRouterConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo router configuration must be a mapping.")
        return MnemoRouterConfig(**config)

    @staticmethod
    def _coerce_delta_config(
        config: dict[str, Any] | MnemoDeltaConfig,
    ) -> MnemoDeltaConfig:
        if isinstance(config, MnemoDeltaConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo delta configuration must be a mapping.")
        return MnemoDeltaConfig(**config)

    @staticmethod
    def _coerce_safety_config(
        config: dict[str, Any] | MnemoSafetyConfig,
    ) -> MnemoSafetyConfig:
        if isinstance(config, MnemoSafetyConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo safety configuration must be a mapping.")
        return MnemoSafetyConfig(**config)
