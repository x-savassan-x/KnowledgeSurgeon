from ...config import MnemoConfig
from ..editor_utils import EditRequest
from ..r_rome.r_rome_main import apply_r_rome_to_model


class MnemoOnlinePhase:
    def __init__(self, model, tokenizer, config: MnemoConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @property
    def weight_names(self) -> list[str]: ...

    def apply(self, request: EditRequest): ...


class RRomeOnlinePhase(MnemoOnlinePhase):
    @property
    def weight_names(self) -> list[str]:
        return [
            f"{self.config.online_config.rewrite_module_tmp.format(layer)}.weight"
            for layer in self.config.online_config.layers
        ]

    def apply(self, request: EditRequest):
        apply_r_rome_to_model(
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


def build_online_phase(model, tokenizer, config: MnemoConfig) -> MnemoOnlinePhase:
    return RRomeOnlinePhase(model, tokenizer, config)
