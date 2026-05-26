from abc import ABC, abstractmethod

from ..edtior_utils import EditRequest
from ...config import MnemoConfig
from ..memit.memit_main import apply_memit_to_model


class MnemoBatchPhase(ABC):
    def __init__(self, model, tokenizer, config: MnemoConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def apply(self, requests: list[EditRequest]): ...


class MemitBatchPhase(MnemoBatchPhase):
    def apply(self, requests: list[EditRequest]):
        apply_memit_to_model(
            self.model,
            self.tokenizer,
            [self._to_backend_request(request) for request in requests],
            self.config.batch_config,
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


BATCH_PHASE_REGISTRY: dict[str, type[MnemoBatchPhase]] = {
    "MEMIT": MemitBatchPhase,
}


def build_batch_phase(model, tokenizer, config: MnemoConfig) -> MnemoBatchPhase:
    alg_name = config.batch_config.alg_name
    if alg_name not in BATCH_PHASE_REGISTRY:
        raise ValueError(f"Unsupported Mnemo batch phase algorithm: {alg_name}.")
    return BATCH_PHASE_REGISTRY[alg_name](model, tokenizer, config)
