from abc import ABC, abstractmethod
from dataclasses import asdict

from .edtior_utils import EditRequest, load_model_and_tokenizer
from ..config import Config


class Editor(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = load_model_and_tokenizer(**asdict(config))
        self._is_last_edit_sequential = False

    def edit(self, request: EditRequest, is_sequential=False):
        if not self._is_last_edit_sequential:
            self._revert()
        self._apply(request)
        if is_sequential:
            self._update_checkpoint()
        self._is_last_edit_sequential = is_sequential

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def _apply(self, request: EditRequest): ...

    @abstractmethod
    def _update_checkpoint(self): ...

    @abstractmethod
    def _revert(self): ...
