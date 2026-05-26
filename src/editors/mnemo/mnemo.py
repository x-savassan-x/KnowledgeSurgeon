import torch

from ..editor import Editor
from ..edtior_utils import EditRequest
from ...config import MnemoConfig
from .batch import build_batch_phase
from .online import build_online_phase


class Mnemo(Editor):
    config: MnemoConfig

    def __init__(self, config: MnemoConfig, model=None, tokenizer=None):
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "right"

        self.online_phase = build_online_phase(self.model, self.tokenizer, config)
        self.batch_phase = build_batch_phase(self.model, self.tokenizer, config)

        self._buffer: list[EditRequest] = []
        self._cycle_start_online_weights: dict[str, torch.Tensor] = {}

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def edit(self, request: EditRequest):
        if not self._cycle_start_online_weights:
            self._snapshot_cycle_start_online_weights()

        self._buffer.append(request)
        self.online_phase.apply(request)

        if self.config.auto_consolidate and self.buffer_size >= self.config.buffer_capacity:
            self.consolidate()

    def consolidate(self):
        if not self._buffer:
            return

        self._restore_cycle_start_online_weights()
        self.batch_phase.apply(self._buffer)

        self._buffer.clear()
        self._cycle_start_online_weights.clear()

    def reset(self):
        self._restore_cycle_start_online_weights()
        self._buffer.clear()
        self._cycle_start_online_weights.clear()

    def _snapshot_cycle_start_online_weights(self):
        params = dict(self.model.named_parameters())
        self._cycle_start_online_weights = {
            name: params[name].detach().clone()
            for name in self.online_phase.weight_names
        }

    def _restore_cycle_start_online_weights(self):
        if not self._cycle_start_online_weights:
            return

        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, weight in self._cycle_start_online_weights.items():
                params[name][...] = weight.clone()
