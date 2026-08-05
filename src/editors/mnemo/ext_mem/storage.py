from dataclasses import dataclass

import numpy as np
import torch

from ...editor_utils import EditRequest
from .router import RoutingDecision, _cpu_float_matrix, _exact_route, _normalize_rows


@dataclass(frozen=True)
class PatchStoreState:
    keys: torch.Tensor
    values: torch.Tensor
    score_threshold: float
    margin_threshold: float


class PatchStore:
    """CPU-backed embedding keys and decoder-block replacement states."""

    def __init__(self, score_threshold: float, margin_threshold: float) -> None:
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)
        self._keys = torch.empty((0, 0), dtype=torch.float32, device="cpu")
        self._values = torch.empty((0, 0), dtype=torch.float32, device="cpu")

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    @property
    def values(self) -> torch.Tensor:
        return self._values

    def append(
        self,
        key: torch.Tensor | np.ndarray,
        value: torch.Tensor,
    ) -> int:
        normalized_key = _cpu_float_matrix(key)
        if normalized_key.ndim == 1:
            normalized_key = normalized_key.unsqueeze(0)
        normalized_key = _normalize_rows(normalized_key)
        if normalized_key.shape[0] != 1:
            raise ValueError("PatchStore.append expects one key")

        value_row = value.detach().to(device="cpu")
        if value_row.ndim == 1:
            value_row = value_row.unsqueeze(0)
        if value_row.ndim != 2 or value_row.shape[0] != 1:
            raise ValueError("PatchStore.append expects one value")

        if len(self):
            if normalized_key.shape[1] != self._keys.shape[1]:
                raise ValueError("Key embedding dimension changed")
            if value_row.shape[1] != self._values.shape[1]:
                raise ValueError("Replacement value dimension changed")
            value_row = value_row.to(dtype=self._values.dtype)
            self._keys = torch.cat((self._keys, normalized_key), dim=0)
            self._values = torch.cat((self._values, value_row), dim=0)
        else:
            self._keys = normalized_key.clone()
            self._values = value_row.clone()
        return len(self) - 1

    def pop(self) -> None:
        if not len(self):
            raise IndexError("Cannot pop from an empty PatchStore")
        self._keys = self._keys[:-1]
        self._values = self._values[:-1]

    def clear(self) -> None:
        key_dimension = self._keys.shape[1] if self._keys.ndim == 2 else 0
        value_dimension = self._values.shape[1] if self._values.ndim == 2 else 0
        value_dtype = self._values.dtype
        self._keys = torch.empty((0, key_dimension), dtype=torch.float32, device="cpu")
        self._values = torch.empty(
            (0, value_dimension),
            dtype=value_dtype,
            device="cpu",
        )

    def copy_state(self) -> PatchStoreState:
        return PatchStoreState(
            keys=self._keys.clone(),
            values=self._values.clone(),
            score_threshold=self.score_threshold,
            margin_threshold=self.margin_threshold,
        )

    def load_state(self, state: PatchStoreState) -> None:
        keys = _normalize_rows(_cpu_float_matrix(state.keys))
        values = state.values.detach().to(device="cpu")
        if keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values):
            raise ValueError("Patch keys and values must be aligned matrices")
        self._keys = keys.clone()
        self._values = values.clone()
        self.score_threshold = float(state.score_threshold)
        self.margin_threshold = float(state.margin_threshold)

    def set_thresholds(self, score_threshold: float, margin_threshold: float) -> None:
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)

    def route(self, query_keys: torch.Tensor | np.ndarray) -> RoutingDecision:
        queries = _cpu_float_matrix(query_keys)
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        queries = _normalize_rows(queries)
        return _exact_route(
            queries,
            self._keys,
            self.score_threshold,
            self.margin_threshold,
        )


@dataclass(frozen=True)
class EditStoreState:
    subjects: tuple[str, ...] | None
    prompts: tuple[str, ...] | None
    answers: tuple[str, ...] | None


class EditStore:
    """Optional raw edit rows aligned with a PatchStore."""

    def __init__(self, compact: bool = False) -> None:
        self._subjects: list[str] | None = None if compact else []
        self._prompts: list[str] | None = None if compact else []
        self._answers: list[str] | None = None if compact else []

    def __len__(self) -> int:
        return 0 if self._subjects is None else len(self._subjects)

    @property
    def compact(self) -> bool:
        return self._subjects is None

    @property
    def subjects(self) -> tuple[str, ...] | None:
        return None if self._subjects is None else tuple(self._subjects)

    @property
    def prompts(self) -> tuple[str, ...] | None:
        return None if self._prompts is None else tuple(self._prompts)

    @property
    def answers(self) -> tuple[str, ...] | None:
        return None if self._answers is None else tuple(self._answers)

    def append(self, request: EditRequest) -> int | None:
        if self.compact:
            return None
        assert self._subjects is not None
        assert self._prompts is not None
        assert self._answers is not None
        self._subjects.append(request.subject)
        self._prompts.append(request.prompt)
        self._answers.append(request.answer)
        return len(self) - 1

    def pop(self) -> None:
        if self.compact or not len(self):
            raise IndexError("Cannot pop from an empty EditStore")
        assert self._subjects is not None
        self._subjects.pop()
        assert self._prompts is not None
        assert self._answers is not None
        self._prompts.pop()
        self._answers.pop()

    def records(self) -> list[EditRequest]:
        if self._prompts is None or self._answers is None:
            raise RuntimeError(
                "Edits compacted for external_only mode cannot be consolidated"
            )
        assert self._subjects is not None
        return [
            EditRequest(subject=subject, prompt=prompt, answer=answer)
            for subject, prompt, answer in zip(
                self._subjects,
                self._prompts,
                self._answers,
                strict=True,
            )
        ]

    def subject_at(self, index: int) -> str | None:
        if self._subjects is None:
            return None
        return self._subjects[index]

    def prompt_at(self, index: int) -> str | None:
        return None if self._prompts is None else self._prompts[index]

    def answer_at(self, index: int) -> str | None:
        return None if self._answers is None else self._answers[index]

    def compact_for_external_only(self) -> None:
        self._subjects = None
        self._prompts = None
        self._answers = None

    def clear(self) -> None:
        if self._subjects is not None:
            self._subjects.clear()
        if self._prompts is not None:
            self._prompts.clear()
        if self._answers is not None:
            self._answers.clear()

    def reset(self, *, compact: bool) -> None:
        self._subjects = None if compact else []
        self._prompts = None if compact else []
        self._answers = None if compact else []

    def copy_state(self) -> EditStoreState:
        return EditStoreState(
            subjects=None if self._subjects is None else tuple(self._subjects),
            prompts=None if self._prompts is None else tuple(self._prompts),
            answers=None if self._answers is None else tuple(self._answers),
        )

    def load_state(self, state: EditStoreState) -> None:
        present = (state.subjects, state.prompts, state.answers)
        if any(value is None for value in present) and any(
            value is not None for value in present
        ):
            raise ValueError("Edit raw fields must be present together")
        if state.subjects is None:
            self._subjects = None
            self._prompts = None
            self._answers = None
            return
        assert state.prompts is not None
        assert state.answers is not None
        if len(state.prompts) != len(state.subjects):
            raise ValueError("Edit prompts must align with subjects")
        if len(state.answers) != len(state.subjects):
            raise ValueError("Edit answers must align with subjects")
        self._subjects = list(state.subjects)
        self._prompts = list(state.prompts)
        self._answers = list(state.answers)


@dataclass(frozen=True)
class MnemoStorageState:
    patches: PatchStoreState
    edits: EditStoreState


class MnemoStorage:
    """Keeps patch tensors aligned with optional raw consolidation rows."""

    def __init__(
        self,
        score_threshold: float,
        margin_threshold: float,
        *,
        compact_edits: bool = False,
    ) -> None:
        self.patches = PatchStore(score_threshold, margin_threshold)
        self.edits = EditStore(compact=compact_edits)

    def __len__(self) -> int:
        self._check_alignment()
        return len(self.patches)

    @property
    def keys(self) -> torch.Tensor:
        return self.patches.keys

    @property
    def values(self) -> torch.Tensor:
        return self.patches.values

    @property
    def score_threshold(self) -> float:
        return self.patches.score_threshold

    @property
    def margin_threshold(self) -> float:
        return self.patches.margin_threshold

    def append(
        self,
        key: torch.Tensor | np.ndarray,
        value: torch.Tensor,
        *,
        request: EditRequest,
    ) -> int:
        patch_index = self.patches.append(key, value)
        try:
            edit_index = self.edits.append(request)
        except BaseException:
            self.patches.pop()
            raise
        if edit_index is not None and patch_index != edit_index:
            self.patches.pop()
            self.edits.pop()
            raise RuntimeError("Patch and edit row indices became misaligned")
        return patch_index

    def clear(self) -> None:
        self.patches.clear()
        self.edits.clear()

    def reset(self, *, compact_edits: bool) -> None:
        self.patches.clear()
        self.edits.reset(compact=compact_edits)

    def compact_for_external_only(self) -> None:
        self.edits.compact_for_external_only()

    def set_thresholds(self, score_threshold: float, margin_threshold: float) -> None:
        self.patches.set_thresholds(score_threshold, margin_threshold)

    def route(self, query_keys: torch.Tensor | np.ndarray) -> RoutingDecision:
        return self.patches.route(query_keys)

    def copy_state(self) -> MnemoStorageState:
        self._check_alignment()
        return MnemoStorageState(
            patches=self.patches.copy_state(),
            edits=self.edits.copy_state(),
        )

    def load_state(self, state: MnemoStorageState) -> None:
        self.patches.load_state(state.patches)
        self.edits.load_state(state.edits)
        self._check_alignment()

    def _check_alignment(self) -> None:
        if self.edits.compact:
            return
        if len(self.patches) != len(self.edits):
            raise RuntimeError(
                "Mnemo patch and edit stores have different row counts: "
                f"{len(self.patches)} != {len(self.edits)}"
            )
