from abc import ABC, abstractmethod

from ...config import MnemoConfig
from ..editor_utils import EditRequest
from ..rome.rome_main import apply_rome_to_model


class MnemoOnlinePhase(ABC):
    def __init__(self, model, tokenizer, config: MnemoConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @property
    @abstractmethod
    def weight_names(self) -> list[str]: ...

    @abstractmethod
    def apply(self, request: EditRequest): ...


class RomeOnlinePhase(MnemoOnlinePhase):
    @property
    def weight_names(self) -> list[str]:
        return [
            f"{self.config.online_config.rewrite_module_tmp.format(layer)}.weight"
            for layer in self.config.online_config.layers
        ]

    def apply(self, request: EditRequest):
        apply_rome_to_model(
            self.model,
            self.tokenizer,
            [self._to_backend_request(request)],
            self.config.online_config,
            return_orig_weights=False,
        )

    @staticmethod
    def _to_backend_request(request: EditRequest) -> dict[str, str]:
        return {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
        }


ONLINE_PHASE_REGISTRY: dict[str, type[MnemoOnlinePhase]] = {
    "ROME": RomeOnlinePhase,
}


def build_online_phase(model, tokenizer, config: MnemoConfig) -> MnemoOnlinePhase:
    alg_name = config.online_config.alg_name
    if alg_name not in ONLINE_PHASE_REGISTRY:
        raise ValueError(f"Unsupported Mnemo online phase algorithm: {alg_name}.")
    return ONLINE_PHASE_REGISTRY[alg_name](model, tokenizer, config)
