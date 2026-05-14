import torch

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import MnemoConfig
from ..memit.memit_main import apply_memit_to_model
from ..rome.rome_main import apply_rome_to_model


class Mnemo(Editor):
    config: MnemoConfig

    def __init__(self, config: MnemoConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "right"
        self._buffer: list[EditRequest] = []
        self._cycle_start_online_weights: dict[str, torch.Tensor] = {}
        self._online_weight_names = self._build_online_weight_names()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def edit(self, request: EditRequest):
        if not self._cycle_start_online_weights:
            self._snapshot_cycle_start_online_weights()

        self._buffer.append(request)
        self._apply_online_edit(request)

        if self.config.auto_consolidate and self.buffer_size >= self.config.buffer_capacity:
            self.consolidate()

    def consolidate(self):
        if not self._buffer:
            return

        self._restore_cycle_start_online_weights()
        self._apply_batch_edit(self._buffer)

        self._buffer.clear()
        self._cycle_start_online_weights.clear()

    def reset(self):
        self._restore_cycle_start_online_weights()
        self._buffer.clear()
        self._cycle_start_online_weights.clear()

    def _build_online_weight_names(self) -> list[str]:
        return [
            f"{self.config.online_config.rewrite_module_tmp.format(layer)}.weight"
            for layer in self.config.online_config.layers
        ]

    def _snapshot_cycle_start_online_weights(self):
        params = dict(self.model.named_parameters())
        self._cycle_start_online_weights = {
            name: params[name].detach().clone()
            for name in self._online_weight_names
        }

    def _restore_cycle_start_online_weights(self):
        if not self._cycle_start_online_weights:
            return

        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._cycle_start_online_weights.items():
                params[name][...] = weight.clone()

    def _apply_online_edit(self, request: EditRequest):
        apply_rome_to_model(
            self.model,
            self.tokenizer,
            [self._to_backend_request(request)],
            self.config.online_config,
            return_orig_weights=False,
        )

    def _apply_batch_edit(self, requests: list[EditRequest]):
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
