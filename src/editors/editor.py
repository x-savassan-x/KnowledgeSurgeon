import gc
import torch

from abc import ABC, abstractmethod
from dataclasses import asdict

from .edtior_utils import EditRequest, load_model_and_tokenizer
from ..config import Config


class Editor(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = load_model_and_tokenizer(**asdict(config))
        self._weights_copy = None


    def __del__(self):
        if hasattr(self, 'model'):
            self.model.cpu()
            del self.model
        self._weights_copy = None
        gc.collect()
        torch.cuda.empty_cache()

    def edit(self, request: EditRequest, is_sequential=False):
        if not is_sequential:
            self._restore()
        self._apply(request)

    @abstractmethod
    def _apply(self, request: EditRequest): ...

    @abstractmethod
    def _restore(self): ...
