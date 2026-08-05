from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config
from ..utils.devices import get_model_device
from .editor_utils import EditRequest
from .warmup import warm_up_model


class Editor(ABC):
    supports_async_generation = True
    async_generation_disabled_reason: str | None = None

    def __init__(
        self,
        config: Config,
        model=None,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.device = config.device

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.config.warmup:
                warm_up_model(
                    self.model,
                    self.tokenizer,
                    model_key=self.config.model_key,
                )
            return

        model_kwargs = {
            "dtype": torch.bfloat16 if config.fp16 else torch.float32,
        }
        if config.model_parallel:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        if not config.model_parallel:
            self.model = self.model.to(self.device)
        self.device = get_model_device(self.model, fallback=config.device)
        self.config.device = self.device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.config.warmup:
            warm_up_model(
                self.model,
                self.tokenizer,
                model_key=self.config.model_key,
            )

    @abstractmethod
    def edit(self, request: EditRequest) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    def drain_log_events(self) -> list[dict[str, object]]:
        """Return and consume JSON-compatible events emitted since the last call."""
        return []

    @abstractmethod
    def copy_for_generation(self, target: "Editor") -> None:
        """Copy the current inference state into a generation-only editor."""
        ...

    @staticmethod
    def copy_named_parameters(
        source: "Editor",
        target: "Editor",
        names: Iterable[str],
    ) -> None:
        """Copy only parameters changed by a parameter-editing method."""
        source_params = dict(source.model.named_parameters())
        target_params = dict(target.model.named_parameters())
        for name in names:
            if name not in source_params or name not in target_params:
                raise KeyError(
                    f"Generation copy cannot find edited parameter {name!r}."
                )
            with torch.no_grad():
                target_params[name].copy_(source_params[name])
        target.model.eval()
