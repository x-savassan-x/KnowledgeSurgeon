from ...config import MnemoConfig
from ..editor import Editor
from ..editor_utils import EditRequest


class Mnemo(Editor):
    """Placeholder for the archived Mnemo prototype."""

    config: MnemoConfig

    def __init__(self, config: MnemoConfig, model=None, tokenizer=None) -> None:
        raise NotImplementedError(
            "Mnemo is archived while its replacement is being designed. "
            "The previous implementation is available under src_orig/mnemo."
        )

    def edit(self, request: EditRequest) -> None:
        raise NotImplementedError("Mnemo does not currently have an active implementation.")

    def reset(self) -> None:
        raise NotImplementedError("Mnemo does not currently have an active implementation.")
