import copy

import torch

from ...config import WiseConfig
from ..editor import Editor
from ..editor_utils import EditRequest
from .wise_adapter import EditingMeanAct, WISEAdapter, WISEModule
from .wise_main import apply_wise_to_model


class WISE(Editor):
    config: WiseConfig

    def __init__(self, config: WiseConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "left"
        self.config.sequential_edit = True
        self._wise_editor: WISEModule | None = None

    def reset(self) -> None:
        if self._wise_editor is not None:
            self._wise_editor.reset_layer()
            self._wise_editor = None
 
    def edit(self, request: EditRequest) -> None:
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.new_ground_truth,
            "loc_prompt": request.subject,
        }]

        self._wise_editor, _ = apply_wise_to_model(
            self.model, self.tokenizer, ee_request, self.config,
        )

    def sync_generation_state_to(self, target: "WISE") -> None:
        self._ensure_generation_adapter(target)
        target.model.load_state_dict(self.model.state_dict(), strict=True)

        source_adapter = self._get_adapter()
        target_adapter = target._get_adapter()
        if source_adapter is not None and target_adapter is not None:
            self._copy_adapter_state(source_adapter, target_adapter)

        target.model.eval()

    def _ensure_generation_adapter(self, target: "WISE") -> None:
        if self._wise_editor is None:
            if target._wise_editor is not None:
                target.reset()
            return

        if target._wise_editor is None:
            target._wise_editor = WISEModule(
                config=target.config,
                model=target.model,
                device=target.device,
            )

    def _get_adapter(self) -> WISEAdapter | None:
        if self._wise_editor is None:
            return None
        adapter = self._wise_editor.get_adapter_layer()
        return adapter if isinstance(adapter, WISEAdapter) else None

    @staticmethod
    def _clone_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
        return value.detach().to(device).clone()

    @classmethod
    def _clone_parameter(cls, value: torch.Tensor, device: torch.device) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            cls._clone_tensor(value, device),
            requires_grad=getattr(value, "requires_grad", False),
        )

    @classmethod
    def _copy_adapter_state(cls, source: WISEAdapter, target: WISEAdapter) -> None:
        device = target.device

        target.new_weight = cls._clone_parameter(source.new_weight, device)
        target.memory_weight = [
            cls._clone_tensor(weight, device)
            for weight in getattr(source, "memory_weight", [])
        ]
        target.memory_mean_act = [
            copy.deepcopy(item)
            for item in getattr(source, "memory_mean_act", [])
        ]
        target.editing_mean_act = copy.deepcopy(
            getattr(source, "editing_mean_act", EditingMeanAct())
        )
        target.editing_total_cnt = getattr(source, "editing_total_cnt", 0)
        target.merge_cnt = getattr(source, "merge_cnt", 0)

        if hasattr(source, "weight_mask"):
            target.weight_mask = cls._clone_tensor(source.weight_mask, device)
        elif hasattr(target, "weight_mask"):
            delattr(target, "weight_mask")

        target.training = False
        target.editing = False
