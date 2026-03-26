from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .edtior_utils import EditRequest
from ..config import Config


class Editor(ABC):
    def __init__(self, config: Config):
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16 if config.fp16 else torch.float32,
        ).cuda(config.gpu_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config = config

    @abstractmethod
    def edit(self, request: EditRequest):...

    @abstractmethod
    def reset(self):...
