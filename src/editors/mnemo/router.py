import copy
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
import inspect

import numpy as np
import torch


@dataclass(frozen=True)
class RoutingDecision:
    top1_indices: torch.Tensor
    top1_scores: torch.Tensor
    top2_scores: torch.Tensor
    margins: torch.Tensor
    active: torch.Tensor

    def __len__(self) -> int:
        return len(self.active)


@dataclass(frozen=True)
class ExternalMemoryState:
    keys: torch.Tensor
    deltas: torch.Tensor
    metadata: list[object]
    score_threshold: float
    margin_threshold: float


class ExternalMemoryBank:
    """CPU-backed key/delta bank with deterministic score-and-margin routing."""

    def __init__(self, score_threshold: float, margin_threshold: float) -> None:
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)
        self._keys = torch.empty((0, 0), dtype=torch.float32, device="cpu")
        self._deltas = torch.empty((0, 0), dtype=torch.float32, device="cpu")
        self._metadata: list[object] = []

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    @property
    def deltas(self) -> torch.Tensor:
        return self._deltas

    @property
    def metadata(self) -> list[object]:
        return self._metadata

    def add(
        self,
        key: torch.Tensor | np.ndarray,
        delta: torch.Tensor,
        metadata: object = None,
    ) -> int:
        normalized_key = _cpu_float_matrix(key)
        if normalized_key.ndim == 1:
            normalized_key = normalized_key.unsqueeze(0)
        normalized_key = _normalize_rows(normalized_key)
        if normalized_key.shape[0] != 1:
            raise ValueError("ExternalMemoryBank.add expects one key")

        delta_row = delta.detach().to(device="cpu")
        if delta_row.ndim == 1:
            delta_row = delta_row.unsqueeze(0)
        if delta_row.ndim != 2 or delta_row.shape[0] != 1:
            raise ValueError("ExternalMemoryBank.add expects one delta")

        if len(self):
            if normalized_key.shape[1] != self._keys.shape[1]:
                raise ValueError("Key embedding dimension changed")
            if delta_row.shape[1] != self._deltas.shape[1]:
                raise ValueError("Delta hidden dimension changed")
            delta_row = delta_row.to(dtype=self._deltas.dtype)
            self._keys = torch.cat((self._keys, normalized_key), dim=0)
            self._deltas = torch.cat((self._deltas, delta_row), dim=0)
        else:
            self._keys = normalized_key.clone()
            self._deltas = delta_row.clone()
        self._metadata.append(copy.deepcopy(metadata))
        return len(self) - 1

    def clear(self) -> None:
        key_dimension = self._keys.shape[1] if self._keys.ndim == 2 else 0
        delta_dimension = self._deltas.shape[1] if self._deltas.ndim == 2 else 0
        delta_dtype = self._deltas.dtype
        self._keys = torch.empty((0, key_dimension), dtype=torch.float32, device="cpu")
        self._deltas = torch.empty((0, delta_dimension), dtype=delta_dtype, device="cpu")
        self._metadata.clear()

    def copy_state(self) -> ExternalMemoryState:
        return ExternalMemoryState(
            keys=self._keys.clone(),
            deltas=self._deltas.clone(),
            metadata=copy.deepcopy(self._metadata),
            score_threshold=self.score_threshold,
            margin_threshold=self.margin_threshold,
        )

    def load_state(self, state: ExternalMemoryState) -> None:
        keys = _normalize_rows(_cpu_float_matrix(state.keys))
        deltas = state.deltas.detach().to(device="cpu")
        if keys.ndim != 2 or deltas.ndim != 2 or len(keys) != len(deltas):
            raise ValueError("External memory keys and deltas must be aligned matrices")
        if len(state.metadata) != len(keys):
            raise ValueError("External memory metadata must align with keys")
        self._keys = keys.clone()
        self._deltas = deltas.clone()
        self._metadata = copy.deepcopy(state.metadata)
        self.score_threshold = float(state.score_threshold)
        self.margin_threshold = float(state.margin_threshold)

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


class BgeM3Router:
    """Dense BGE-M3 router with an exact cosine score-and-margin gate."""

    def __init__(
        self,
        score_threshold: float,
        margin_threshold: float,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        max_length: int = 384,
        batch_size: int = 64,
        embedder=None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)
        self.model_name = model_name
        self.device = str(device)
        self.use_fp16 = bool(use_fp16 and self.device.startswith("cuda"))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._embedder = embedder
        self._keys = torch.empty((0, 0), dtype=torch.float32, device="cpu")

    @property
    def bank_size(self) -> int:
        return len(self._keys)

    @property
    def embedding_dimension(self) -> int | None:
        return None if self._keys.ndim != 2 or self._keys.shape[1] == 0 else self._keys.shape[1]

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    def clear(self) -> None:
        dimension = self._keys.shape[1] if self._keys.ndim == 2 else 0
        self._keys = torch.empty((0, dimension), dtype=torch.float32, device="cpu")

    def encode(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            dimension = self.embedding_dimension or 0
            return torch.empty((0, dimension), dtype=torch.float32, device="cpu")

        embedder = self._get_embedder()
        encode = embedder.encode
        kwargs = {
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }
        try:
            signature = inspect.signature(encode)
        except (TypeError, ValueError):
            call_kwargs = kwargs
        else:
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            call_kwargs = (
                kwargs
                if accepts_kwargs
                else {key: value for key, value in kwargs.items() if key in signature.parameters}
            )

        with redirect_stderr(StringIO()):
            encoded = encode(prompts, **call_kwargs)
        if isinstance(encoded, dict):
            if "dense_vecs" not in encoded:
                raise ValueError("Embedding result does not contain dense_vecs")
            encoded = encoded["dense_vecs"]

        embeddings = _cpu_float_matrix(encoded)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"Unexpected embedding shape {tuple(embeddings.shape)} for {len(prompts)} prompts"
            )
        return _normalize_rows(embeddings)

    def add_prompts(self, prompts: list[str]) -> torch.Tensor:
        embeddings = self.encode(prompts)
        self.add_embeddings(embeddings)
        return embeddings

    def add_embeddings(self, embeddings: torch.Tensor | np.ndarray) -> None:
        normalized = _normalize_rows(_cpu_float_matrix(embeddings))
        if normalized.ndim != 2:
            raise ValueError("Router keys must be a 2D matrix")
        if self.bank_size and normalized.shape[1] != self._keys.shape[1]:
            raise ValueError(
                f"Embedding dimension changed from {self._keys.shape[1]} to {normalized.shape[1]}"
            )
        if not self.bank_size:
            self._keys = normalized.clone()
        elif len(normalized):
            self._keys = torch.cat((self._keys, normalized), dim=0)

    def load_keys(self, keys: torch.Tensor | np.ndarray) -> None:
        normalized = _normalize_rows(_cpu_float_matrix(keys))
        if normalized.ndim != 2:
            raise ValueError("Router keys must be a 2D matrix")
        self._keys = normalized.clone()

    def retrieve_prompts(self, prompts: list[str]) -> RoutingDecision:
        if not prompts:
            return self._empty_decision(0)
        if self.bank_size == 0:
            return self._empty_decision(len(prompts))
        return self.retrieve_embeddings(self.encode(prompts))

    def retrieve_embeddings(
        self,
        embeddings: torch.Tensor | np.ndarray,
    ) -> RoutingDecision:
        queries = _normalize_rows(_cpu_float_matrix(embeddings))
        if queries.ndim != 2:
            raise ValueError("Router queries must be a 2D matrix")
        if self.bank_size == 0:
            return self._empty_decision(len(queries))
        if queries.shape[1] != self._keys.shape[1]:
            raise ValueError(
                f"Query dimension {queries.shape[1]} does not match key dimension {self._keys.shape[1]}"
            )

        return _exact_route(
            queries,
            self._keys,
            self.score_threshold,
            self.margin_threshold,
        )

    def _get_embedder(self):
        if self._embedder is None:
            from FlagEmbedding import BGEM3FlagModel

            self._embedder = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                devices=self.device,
            )
        return self._embedder

    @staticmethod
    def _empty_decision(count: int) -> RoutingDecision:
        return RoutingDecision(
            top1_indices=torch.full((count,), -1, dtype=torch.long),
            top1_scores=torch.full((count,), -torch.inf, dtype=torch.float32),
            top2_scores=torch.full((count,), -torch.inf, dtype=torch.float32),
            margins=torch.full((count,), -torch.inf, dtype=torch.float32),
            active=torch.zeros(count, dtype=torch.bool),
        )


def _cpu_float_matrix(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device="cpu")


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        return values
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("Embedding vectors must have non-zero norm")
    return values / norms.clamp_min(1e-12)


def _exact_route(
    queries: torch.Tensor,
    keys: torch.Tensor,
    score_threshold: float,
    margin_threshold: float,
) -> RoutingDecision:
    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError("Router queries and keys must be 2D matrices")
    if len(keys) == 0:
        return BgeM3Router._empty_decision(len(queries))
    if queries.shape[1] != keys.shape[1]:
        raise ValueError(
            f"Query dimension {queries.shape[1]} does not match key dimension {keys.shape[1]}"
        )

    scores = queries @ keys.T
    if len(keys) == 1:
        top1_scores = scores[:, 0]
        top1_indices = torch.zeros(len(queries), dtype=torch.long)
        top2_scores = torch.full_like(top1_scores, -torch.inf)
        margins = torch.full_like(top1_scores, torch.inf)
    else:
        top_scores, top_indices = torch.topk(scores, k=2, dim=1)
        top1_scores = top_scores[:, 0]
        top2_scores = top_scores[:, 1]
        top1_indices = top_indices[:, 0]
        margins = top1_scores - top2_scores

    active = (
        (top1_scores >= score_threshold)
        & (margins >= margin_threshold)
    )
    return RoutingDecision(
        top1_indices=top1_indices,
        top1_scores=top1_scores,
        top2_scores=top2_scores,
        margins=margins,
        active=active,
    )
