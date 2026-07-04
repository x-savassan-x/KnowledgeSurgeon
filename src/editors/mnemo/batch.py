from abc import ABC, abstractmethod

from ...config import MnemoConfig
from ..editor_utils import EditRequest
from ..alpha_edit import alpha_edit_main as _ae_mod
from ..alpha_edit.alpha_edit_main import apply_AlphaEdit_to_model
from ..memit.memit_main import apply_memit_to_model


class MnemoBatchPhase(ABC):
    def __init__(self, model, tokenizer, config: MnemoConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @property
    @abstractmethod
    def weight_names(self) -> list[str]: ...

    @abstractmethod
    def apply(self, requests: list[EditRequest]): ...

    def reset(self):
        pass

    @staticmethod
    def _to_backend_request(request: EditRequest) -> dict[str, str]:
        return {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
        }


class MemitBatchPhase(MnemoBatchPhase):
    @property
    def weight_names(self) -> list[str]:
        return [
            f"{self.config.batch_config.rewrite_module_tmp.format(layer)}.weight"
            for layer in self.config.batch_config.layers
        ]

    def apply(self, requests: list[EditRequest]):
        apply_memit_to_model(
            self.model,
            self.tokenizer,
            [self._to_backend_request(request) for request in requests],
            self.config.batch_config,
            return_orig_weights=False,
        )


class AlphaEditBatchPhase(MnemoBatchPhase):
    @property
    def weight_names(self) -> list[str]:
        return [
            f"{self.config.batch_config.rewrite_module_tmp.format(layer)}.weight"
            for layer in self.config.batch_config.layers
        ]

    def apply(self, requests: list[EditRequest]):
        apply_AlphaEdit_to_model(
            self.model,
            self.tokenizer,
            [self._to_backend_request(request) for request in requests],
            self.config.batch_config,
            return_orig_weights=False,
        )

    def reset(self):
        if hasattr(_ae_mod, "cache_c"):
            _ae_mod.cache_c = None
        _ae_mod.cache_c_new = False


BATCH_PHASE_REGISTRY: dict[str, type[MnemoBatchPhase]] = {
    "MEMIT": MemitBatchPhase,
    "AlphaEdit": AlphaEditBatchPhase,
}


def build_batch_phase(model, tokenizer, config: MnemoConfig) -> MnemoBatchPhase:
    alg_name = config.batch_config.alg_name
    if alg_name not in BATCH_PHASE_REGISTRY:
        raise ValueError(f"Unsupported Mnemo batch phase algorithm: {alg_name}.")
    return BATCH_PHASE_REGISTRY[alg_name](model, tokenizer, config)
