from dataclasses import dataclass, field
from typing import Any

from ..config import Config, resolve_project_path


# The global experiment calibrates through 5,000 edits. Each null fallback
# deliberately reuses that last profile; it is not an infinite-bank calibration.
MNEMO_ROUTER_PRESETS: dict[str, dict[str, Any]] = {
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "thresholds": (
            (100, 0.6630830764770508, 0.07978279143571854),
            (1_000, 0.7187305092811584, 0.07721257954835892),
            (2_000, 0.7015174031257629, 0.0976194217801094),
            (5_000, 0.7373771071434021, 0.06535560637712479),
            (None, 0.7373771071434021, 0.06535560637712479),
        ),
    },
    "e5-base-v2": {
        "model_name": "intfloat/e5-base-v2",
        "thresholds": (
            (100, 0.8250033259391785, 0.044400040060281754),
            (1_000, 0.8329248428344727, 0.0455482043325901),
            (2_000, 0.843231737613678, 0.03516644611954689),
            (5_000, 0.8472376465797424, 0.0338285006582737),
            (None, 0.8472376465797424, 0.0338285006582737),
        ),
    },
    "all-minilm-l6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "thresholds": (
            (100, 0.5154905915260315, 0.1807645708322525),
            (1_000, 0.5296052694320679, 0.1870354264974594),
            (2_000, 0.6048957705497742, 0.10272795706987381),
            (5_000, 0.5874411463737488, 0.12655426561832428),
            (None, 0.5874411463737488, 0.12655426561832428),
        ),
    },
}


@dataclass
class MnemoDiagnosticsConfig:
    # Global override; null delegates control to the component switches.
    all: bool | None = None

    # Report intervention-vector loss, norm, and saturation after each edit.
    delta: bool = False

    # Report retrieval scores, gates, selected memory rows, and route outcomes.
    router: bool = False

    # Report whether the decoder-block hook observed and modified each batch row.
    interceptor: bool = False

    def __post_init__(self) -> None:
        if self.all is not None and not isinstance(self.all, bool):
            raise TypeError("Mnemo diagnostics all must be a boolean or null.")
        for name in ("delta", "router", "interceptor"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Mnemo diagnostics {name} must be a boolean.")

    @property
    def delta_enabled(self) -> bool:
        return self.delta if self.all is None else self.all

    @property
    def router_enabled(self) -> bool:
        return self.router if self.all is None else self.all

    @property
    def interceptor_enabled(self) -> bool:
        return self.interceptor if self.all is None else self.all


@dataclass
class MnemoRouterThresholdConfig:
    # Largest active memory bank covered by this calibration; null is the fallback.
    max_bank_size: int | None

    # Minimum top-1 cosine similarity required to activate retrieval.
    score_threshold: float

    # Minimum difference between the top-1 and top-2 similarity scores.
    margin_threshold: float

    def __post_init__(self) -> None:
        if self.max_bank_size is not None:
            if isinstance(self.max_bank_size, bool) or not isinstance(
                self.max_bank_size, int
            ):
                raise TypeError("Mnemo router max_bank_size must be an integer or null.")
            if self.max_bank_size <= 0:
                raise ValueError("Mnemo router max_bank_size must be positive.")
        if not -1.0 <= self.score_threshold <= 1.0:
            raise ValueError("Mnemo router score_threshold must be in [-1, 1].")
        if not 0.0 <= self.margin_threshold <= 2.0:
            raise ValueError("Mnemo router margin_threshold must be in [0, 2].")


@dataclass
class MnemoRouterConfig:
    # Calibrated encoder preset: bge-base-en-v1.5, e5-base-v2, or all-minilm-l6-v2.
    backend: str = "bge-base-en-v1.5"

    # Hugging Face model ID fixed by the calibrated encoder preset.
    model_name: str = field(init=False)

    # Device used for prompt encoding, or auto to select one at runtime.
    device: str = "auto"

    # Run supported prompt encoders in float16 on CUDA.
    fp16: bool = True

    # Load the prompt encoder during Mnemo initialization instead of first use.
    eager_load: bool = True

    # Maximum token length accepted by the prompt encoder.
    max_length: int = 384

    # Number of prompts encoded per inference batch.
    batch_size: int = 32

    # Retrieval gate applied to the nearest-neighbor scores.
    gate: str = "score_margin"

    # Optional capacity calibration override; null uses the selected preset.
    thresholds: list[
        dict[str, float | int | None] | MnemoRouterThresholdConfig
    ] | None = None

    threshold_configs: tuple[MnemoRouterThresholdConfig, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("Mnemo router backend must not be empty.")
        self.backend = self.backend.strip().casefold().replace("_", "-")
        preset = MNEMO_ROUTER_PRESETS.get(self.backend)
        if preset is None:
            supported = ", ".join(MNEMO_ROUTER_PRESETS)
            raise ValueError(
                f"Unsupported Mnemo router backend {self.backend!r}. "
                f"Supported backends: {supported}."
            )
        self.model_name = str(preset["model_name"])
        if self.thresholds is None:
            self.thresholds = [
                {
                    "max_bank_size": max_bank_size,
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                }
                for max_bank_size, score_threshold, margin_threshold in preset["thresholds"]
            ]
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
        if not self.thresholds:
            raise ValueError("Mnemo router thresholds must not be empty.")
        profiles = tuple(self._coerce_thresholds(config) for config in self.thresholds)
        bounded_sizes = [
            profile.max_bank_size
            for profile in profiles
            if profile.max_bank_size is not None
        ]
        if bounded_sizes != sorted(set(bounded_sizes)):
            raise ValueError(
                "Mnemo router bounded max_bank_size values must be unique and increasing."
            )
        fallback_indices = [
            index
            for index, profile in enumerate(profiles)
            if profile.max_bank_size is None
        ]
        if fallback_indices != [len(profiles) - 1]:
            raise ValueError(
                "Mnemo router thresholds require one null max_bank_size fallback last."
            )
        self.threshold_configs = profiles

    @staticmethod
    def _coerce_thresholds(
        config: dict[str, float | int | None] | MnemoRouterThresholdConfig,
    ) -> MnemoRouterThresholdConfig:
        if isinstance(config, MnemoRouterThresholdConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo router thresholds must be a mapping.")
        return MnemoRouterThresholdConfig(**config)

    def profile_for(self, bank_size: int) -> MnemoRouterThresholdConfig:
        if isinstance(bank_size, bool) or not isinstance(bank_size, int):
            raise TypeError("Mnemo router bank_size must be an integer.")
        if bank_size < 0:
            raise ValueError("Mnemo router bank_size must be nonnegative.")
        return next(
            profile
            for profile in self.threshold_configs
            if profile.max_bank_size is None or bank_size <= profile.max_bank_size
        )


@dataclass
class MnemoDeltaConfig:
    # Maximum optimization steps used to learn each intervention vector.
    num_steps: int = 50

    # Learning rate for intervention-vector optimization.
    learning_rate: float = 1.0

    # L2 weight decay applied to the intervention vector.
    weight_decay: float = 0.0

    # Maximum L2 norm allowed for an intervention vector.
    max_norm: float = 40.0

    # Stop optimization once the target loss reaches this value.
    early_stop_loss: float = 1.0e-3

    # Prefix targets with a space before tokenization when appropriate.
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
    # Enable behavioral, retention, and SPHERE checks; the numerical guard always runs.
    enabled: bool = True

    # Number of consolidated edits before full safety checks begin.
    start_after_edits: int = 500

    # Number of validation prompts evaluated per forward-pass microbatch.
    validation_microbatch_size: int = 4

    # Neutral prompts used to measure behavioral and geometric drift.
    canary_prompts: list[str] = field(default_factory=_default_canary_prompts)

    # Number of canary prompts sampled for each safety check.
    canary_sample_size: int = 16

    # Maximum number of prior edits retained for success checks.
    retention_reservoir_size: int = 256

    # Number of retained edits sampled for each safety check.
    retention_sample_size: int = 64

    # Number of canary activations sampled for SPHERE energy checks.
    sphere_sample_size: int = 64

    # Row chunk size used when computing pairwise SPHERE energy.
    sphere_chunk_size: int = 16

    # Seed used for deterministic SPHERE activation sampling.
    sphere_seed: int = 0

    # Exponent applied to inverse pairwise distance in SPHERE energy.
    sphere_exponent: float = 1.0

    # Maximum tensor elements inspected at once by numerical guards.
    tensor_chunk_elements: int = 1_000_000

    # Number of warning-level checks required to reject consolidation.
    warning_votes_required: int = 2

    # Relative update norm that raises a numerical warning.
    relative_update_norm_warning: float = 0.10

    # Relative update norm that rejects consolidation immediately.
    relative_update_norm_hard_limit: float = 1.0

    # Canary KL drift from the last safe model that raises a warning.
    step_kl_warning: float = 0.05

    # Canary KL drift from the last safe model that rejects consolidation.
    step_kl_hard_limit: float = 0.20

    # Canary KL drift from the original model that raises a warning.
    cumulative_kl_warning: float = 0.10

    # Canary KL drift from the original model that rejects consolidation.
    cumulative_kl_hard_limit: float = 0.50

    # Canary NLL increase from the last safe model that raises a warning.
    step_nll_warning: float = 0.25

    # Canary NLL increase from the last safe model that rejects consolidation.
    step_nll_hard_limit: float = 0.75

    # Canary NLL increase from the original model that raises a warning.
    cumulative_nll_warning: float = 0.50

    # Canary NLL increase from the original model that rejects consolidation.
    cumulative_nll_hard_limit: float = 1.50

    # Minimum success rate required on the consolidation's current edit batch.
    current_batch_min_success: float = 0.80

    # Minimum success rate required on sampled retained edits.
    retention_min_success: float = 0.80

    # Maximum retained-success drop relative to the last safe model.
    retention_max_mean_drop: float = 0.20

    # Relative SPHERE energy drift from the last safe model that warns.
    sphere_step_warning: float = 0.10

    # Relative SPHERE energy drift from the last safe model that rejects.
    sphere_step_hard_limit: float = 0.25

    # Relative SPHERE energy drift from the original model that warns.
    sphere_cumulative_warning: float = 0.20

    # Relative SPHERE energy drift from the original model that rejects.
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
class MnemoConsolidationConfig:
    # Transformer layers whose MLP weights receive projected updates.
    layers: list[int] = field(default_factory=list)

    # Subject token position used to extract the rewrite key.
    fact_token: str = "subject_last"

    # Maximum optimization steps used to solve each target value vector.
    v_num_grad_steps: int = 20

    # Learning rate for target value optimization.
    v_lr: float = 5e-1

    # Layer where the target prediction loss is measured.
    v_loss_layer: int = 47

    # Weight decay applied to optimized target values.
    v_weight_decay: float = 0.5

    # Maximum target-value norm relative to the original activation norm.
    clamp_norm_factor: float = 0.75

    # Weight assigned to the KL preservation loss.
    kl_factor: float = 0.0625

    # Dot-path template for the MLP weight rewritten at each layer.
    rewrite_module_tmp: str = ""

    # Dot-path template for a transformer layer.
    layer_module_tmp: str = ""

    # Dot path for the model's final normalization module.
    ln_f_module: str = ""

    # Dot path for the model's language-model head.
    lm_head_module: str = ""

    # Dataset used to estimate the activation second moment.
    mom2_dataset: str = "wikipedia"

    # Number of samples used to estimate the activation second moment.
    mom2_n_samples: int = 100000

    # Storage dtype for activation statistics.
    mom2_dtype: str = "float32"

    # Directory containing cached activation statistics.
    stats_dir: str = "data/assets/layer_stats"

    # Eigenvalue cutoff used to construct the protected null space.
    nullspace_threshold: float = 2e-2

    # L2 regularization coefficient used when solving weight updates.
    L2: float = 10

    # Path to the cached null-space projector.
    P_loc: str = "data/assets/alpha_edit_projectors/null_space_project.pt"

    # Runtime device supplied by Mnemo after model initialization.
    device: Any = field(default=None, init=False, repr=False)

    # Stable model slug used to separate cached activation statistics.
    model_key: str = field(default="", init=False, repr=False)

    # Optional token limit for activation statistics.
    layer_stats_max_length: int | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Mnemo consolidation layers must not be empty.")
        if self.v_num_grad_steps <= 0:
            raise ValueError("Mnemo consolidation v_num_grad_steps must be positive.")
        if self.layer_stats_max_length is not None and self.layer_stats_max_length <= 0:
            raise ValueError(
                "Mnemo consolidation layer_stats_max_length must be positive when set."
            )
        self.stats_dir = resolve_project_path(self.stats_dir)
        self.P_loc = resolve_project_path(self.P_loc)


@dataclass
class MnemoConfig(Config):
    # Number of externally stored edits consolidated in each weight update.
    consolidation_cycle: int = 100

    # Start in external_only mode instead of hybrid mode.
    external_only: bool = False

    # Component diagnostics with an optional all-on/all-off override.
    diagnostics: dict[str, bool | None] | MnemoDiagnosticsConfig = field(
        default_factory=dict
    )

    # Decoder block index where retrieved interventions are applied.
    intervention_block: int = 18

    # Dot-path template for the decoder block intercepted by Mnemo.
    intervention_module_tmp: str = "model.layers.{}"

    # Blend strength applied to a retrieved intervention vector.
    intervention_strength: float = 1.0

    # Mnemo-owned projected weight writer used in hybrid mode.
    consolidation: dict[str, Any] | MnemoConsolidationConfig = field(
        default_factory=dict
    )

    # Prompt encoder, retrieval gate, and threshold configuration.
    router: dict[str, Any] | MnemoRouterConfig = field(default_factory=dict)

    # Optimization settings for externally stored intervention vectors.
    delta: dict[str, Any] | MnemoDeltaConfig = field(default_factory=dict)

    # Validation settings used before committing consolidation updates.
    safety: dict[str, Any] | MnemoSafetyConfig = field(default_factory=dict)

    consolidation_config: MnemoConsolidationConfig = field(init=False)
    router_config: MnemoRouterConfig = field(init=False)
    delta_config: MnemoDeltaConfig = field(init=False)
    safety_config: MnemoSafetyConfig = field(init=False)
    diagnostics_config: MnemoDiagnosticsConfig = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.consolidation_cycle <= 0:
            raise ValueError("Mnemo consolidation_cycle must be positive.")
        if not isinstance(self.external_only, bool):
            raise TypeError("Mnemo external_only must be a boolean.")
        if self.intervention_block < 0:
            raise ValueError("Mnemo intervention_block must be nonnegative.")
        if "{}" not in self.intervention_module_tmp:
            raise ValueError("Mnemo intervention_module_tmp must contain '{}'.")
        if not 0.0 < self.intervention_strength <= 1.0:
            raise ValueError("Mnemo intervention_strength must be in (0, 1].")
        self.consolidation_config = self._coerce_consolidation_config(
            self.consolidation
        )
        self.consolidation_config.device = self.device
        self.consolidation_config.model_key = self.model_key
        self.consolidation_config.layer_stats_max_length = self.layer_stats_max_length
        self.router_config = self._coerce_router_config(self.router)
        self.delta_config = self._coerce_delta_config(self.delta)
        self.safety_config = self._coerce_safety_config(self.safety)
        self.diagnostics_config = self._coerce_diagnostics_config(self.diagnostics)

    @staticmethod
    def _coerce_diagnostics_config(
        config: dict[str, bool | None] | MnemoDiagnosticsConfig,
    ) -> MnemoDiagnosticsConfig:
        if isinstance(config, MnemoDiagnosticsConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo diagnostics configuration must be a mapping.")
        return MnemoDiagnosticsConfig(**config)

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

    @staticmethod
    def _coerce_consolidation_config(
        config: dict[str, Any] | MnemoConsolidationConfig,
    ) -> MnemoConsolidationConfig:
        if isinstance(config, MnemoConsolidationConfig):
            return config
        if not isinstance(config, dict):
            raise TypeError("Mnemo consolidation configuration must be a mapping.")
        return MnemoConsolidationConfig(**config)
