import copy

import torch

from ...config import GraceConfig
from ..editor import Editor
from ..edtior_utils import EditRequest
from .grace_core import GraceAdapter, GraceCore
from .grace_main import apply_grace_to_model
from .utils import brackets_to_periods, parent_module, tokenize


class GRACE(Editor):
    config: GraceConfig

    def __init__(self, config: GraceConfig, model=None, tokenizer=None):
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "left"
        self._base_model = self.model
        self._grace_model: GraceCore | None = None
        self._reset_layer = None

    def edit(self, request: EditRequest):
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "target_true": request.old_ground_truth,
            "subject": request.subject,
        }]

        if self._grace_model is None:
            grace_model, reset_layer = apply_grace_to_model(
                self._base_model,
                self.tokenizer,
                ee_request,
                self.config,
                return_orig_weights=True,
            )
            self._grace_model = grace_model
            self.model = grace_model
            self._reset_layer = reset_layer
        else:
            tokens = tokenize(ee_request[0], tokenizer=self.tokenizer, device=self.device)
            self._grace_model.edit(config=self.config, tokens=tokens, edit_id=ee_request[0]["target_new"])

        self._grace_model.set_inference_mode()

    def reset(self):
        if self._reset_layer is not None:
            self._reset_layer()
            self._reset_layer = None
        self._grace_model = None
        self.model = self._base_model

    def sync_generation_state_to(self, target: "GRACE") -> None:
        self._ensure_generation_adapter(target)
        target.model.load_state_dict(self.model.state_dict(), strict=False)

        source_adapter = self._get_adapter()
        target_adapter = target._get_adapter()
        if source_adapter is not None and target_adapter is not None:
            self._copy_adapter_state(source_adapter, target_adapter)
            target._grace_model.set_inference_mode()

        target.model.eval()

    def _ensure_generation_adapter(self, target: "GRACE") -> None:
        if self._grace_model is None:
            if target._grace_model is not None:
                target.reset()
            return

        if target._grace_model is None:
            target._grace_model = GraceCore(
                config=target.config,
                model=target._base_model,
                device=target.device,
            )
            target.model = target._grace_model
            target._reset_layer = target._grace_model.reset_layer

    def _get_adapter(self) -> GraceAdapter | None:
        if self._grace_model is None:
            return None

        layer_name = self._grace_model.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(
            self._grace_model.model,
            brackets_to_periods(self._grace_model.layer),
        )
        adapter = getattr(edit_module, layer_name)
        return adapter if isinstance(adapter, GraceAdapter) else None

    @staticmethod
    def _clone_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        return value.detach().to(device).clone()

    @classmethod
    def _copy_adapter_state(cls, source: GraceAdapter, target: GraceAdapter) -> None:
        device = target.device

        for name in ("keys", "epsilons"):
            if hasattr(source, name):
                setattr(target, name, cls._clone_tensor(getattr(source, name), device))
            elif hasattr(target, name):
                delattr(target, name)

        if hasattr(source, "values"):
            target.values = torch.nn.Parameter(
                cls._clone_tensor(source.values, device),
                requires_grad=source.values.requires_grad,
            )
        elif hasattr(target, "values"):
            delattr(target, "values")

        target.key_labels = [
            cls._clone_tensor(label, device) if isinstance(label, torch.Tensor) else copy.deepcopy(label)
            for label in getattr(source, "key_labels", [])
        ]
        target.edit_ids = copy.deepcopy(getattr(source, "edit_ids", []))
        target.training = False
        target.key_id = -1
