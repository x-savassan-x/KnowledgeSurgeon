import copy
import json
from pathlib import Path
from typing import ClassVar

import torch

from ...config import WiseConfig
from ..editor import Editor
from ..editor_utils import EditRequest
from .wise_adapter import (
    EditingMeanAct,
    WISEAdapter,
    WISEModule,
    clear_edit_histories,
)
from .wise_main import apply_wise_to_model


class WISE(Editor):
    config: WiseConfig
    _loc_prompt_cache: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init__(self, config: WiseConfig, model=None, tokenizer=None) -> None:
        super().__init__(config, model=model, tokenizer=tokenizer)
        self.tokenizer.padding_side = "left"
        self.config.sequential_edit = True
        self._wise_editor: WISEModule | None = None
        self._loc_prompts = self._load_loc_prompts()
        self._loc_index = 0

    def _load_loc_prompts(self) -> tuple[str, ...]:
        path = Path(self.config.loc_data_path).resolve()
        cache_key = str(path)
        cached = self._loc_prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        if not path.exists():
            raise FileNotFoundError(
                "WISE loc_data_path does not exist: "
                f"{path}. Restore the tracked WISE locality bank before running."
            )
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"WISE loc_data_path must contain a JSON list: {path}")
        prompts: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"WISE loc_data row {index} is not an object")
            loc_prompt = record.get("loc")
            loc_answer = record.get("loc_ans")
            if not isinstance(loc_prompt, str) or not isinstance(loc_answer, str):
                raise ValueError(
                    f"WISE loc_data row {index} must contain string loc and loc_ans"
                )
            prompts.append(f"{loc_prompt} {loc_answer}")

        if not prompts:
            raise ValueError(f"WISE loc_data_path contains no usable rows: {path}")
        cached = tuple(prompts)
        self._loc_prompt_cache[cache_key] = cached
        return cached

    def _next_loc_prompt(self) -> str:
        if self._loc_index >= len(self._loc_prompts):
            raise ValueError(
                "WISE loc_data is exhausted; provide at least one irrelevant "
                f"example per edit ({len(self._loc_prompts)} available)."
            )
        loc_prompt = self._loc_prompts[self._loc_index]
        self._loc_index += 1
        return loc_prompt

    def edit(self, request: EditRequest) -> None:
        ee_request = [{
            "prompt": request.prompt,
            "target_new": request.answer,
            "loc_prompt": self._next_loc_prompt(),
        }]

        self._wise_editor, _ = apply_wise_to_model(
            self.model, self.tokenizer, ee_request, self.config,
        )

    def reset(self) -> None:
        if self._wise_editor is not None:
            self._wise_editor.reset_layer()
            self._wise_editor = None
        clear_edit_histories()
        self._loc_index = 0

    def copy_for_generation(self, target: "WISE") -> None:
        if not isinstance(target, WISE):
            raise TypeError("WISE generation copy requires another WISE editor.")

        self._ensure_generation_adapter(target)

        source_adapter = self._get_adapter()
        target_adapter = target._get_adapter()
        if source_adapter is not None and target_adapter is not None:
            # Retrieve mode stores multiple side-memory shards. Merge mode
            # instead accumulates edits in its single side-memory layer.
            if not self.config.retrieve:
                target_adapter.layer.weight = self._clone_parameter(
                    source_adapter.layer.weight,
                    target_adapter.device,
                )
            self._copy_inference_state(source_adapter, target_adapter)

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
    def _copy_inference_state(
        cls,
        source: WISEAdapter,
        target: WISEAdapter,
    ) -> None:
        device = target.device

        with torch.no_grad():
            target.new_weight.copy_(source.new_weight)
        target.new_weight.requires_grad_(False)
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
        if not source.config.retrieve:
            # WISE-Merge may defer its final pending shard merge until inference.
            target.editing_total_cnt = getattr(source, "editing_total_cnt", 0)

        target.training = False
        target.editing = False
