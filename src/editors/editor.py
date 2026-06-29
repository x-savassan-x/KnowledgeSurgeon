from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config
from ..utils.devices import get_model_device
from .edtior_utils import EditRequest


class Editor(ABC):
    def __init__(self, config: Config, model=None, tokenizer=None):
        self.config = config
        self.device = config.device

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            return

        model_kwargs = {
            "dtype": torch.bfloat16 if config.fp16 else torch.float32,
        }
        if config.model_parallel:
            model_kwargs["device_map"] = config.device_map or "auto"

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

    @abstractmethod
    def edit(self, request: EditRequest):...

    @abstractmethod
    def reset(self):...

    def sync_generation_state_to(self, target: "Editor") -> None:
        """Copy the current inference state into a generation-only editor."""
        target.model.load_state_dict(self.model.state_dict(), strict=True)
        target.model.eval()
