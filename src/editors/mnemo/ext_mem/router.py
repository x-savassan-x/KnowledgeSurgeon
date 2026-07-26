from abc import ABC, abstractmethod
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


class PromptEncoder(ABC):
    """Turns prompts into dense embedding rows for a Router."""

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def encode(self, prompts: list[str]) -> torch.Tensor:
        pass


class BgeM3Encoder(PromptEncoder):
    """Dense prompt encoder backed by FlagEmbedding's BGE-M3 model."""

    def __init__(
        self,
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

        self.model_name = model_name
        self.device = str(device)
        self.use_fp16 = bool(use_fp16 and self.device.startswith("cuda"))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._embedder = embedder

    def load(self) -> None:
        self._get_embedder()

    def encode(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            return torch.empty((0, 0), dtype=torch.float32, device="cpu")

        encode = self._get_embedder().encode
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
        return embeddings

    def _get_embedder(self):
        if self._embedder is None:
            from FlagEmbedding import BGEM3FlagModel

            self._embedder = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                devices=self.device,
            )
        return self._embedder


class Router:
    """Exact cosine retrieval with a score-and-margin activation gate."""

    def __init__(
        self,
        encoder: PromptEncoder,
        score_threshold: float,
        margin_threshold: float,
    ) -> None:
        self.encoder = encoder
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)
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

    def load(self) -> None:
        self.encoder.load()

    def set_thresholds(self, score_threshold: float, margin_threshold: float) -> None:
        self.score_threshold = float(score_threshold)
        self.margin_threshold = float(margin_threshold)

    def encode(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            dimension = self.embedding_dimension or 0
            return torch.empty((0, dimension), dtype=torch.float32, device="cpu")

        embeddings = _normalize_rows(_cpu_float_matrix(self.encoder.encode(prompts)))
        if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"Unexpected embedding shape {tuple(embeddings.shape)} for {len(prompts)} prompts"
            )
        return embeddings

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

    @staticmethod
    def _empty_decision(count: int) -> RoutingDecision:
        return RoutingDecision(
            top1_indices=torch.full((count,), -1, dtype=torch.long),
            top1_scores=torch.full((count,), -torch.inf, dtype=torch.float32),
            top2_scores=torch.full((count,), -torch.inf, dtype=torch.float32),
            margins=torch.full((count,), -torch.inf, dtype=torch.float32),
            active=torch.zeros(count, dtype=torch.bool),
        )


ROUTER_ENCODERS: dict[str, type[PromptEncoder]] = {
    "bge-m3": BgeM3Encoder,
}


def build_router(
    *,
    backend: str,
    score_threshold: float,
    margin_threshold: float,
    model_name: str,
    device: str,
    use_fp16: bool,
    max_length: int,
    batch_size: int,
) -> Router:
    backend_key = backend.strip().casefold().replace("_", "-")
    encoder_type = ROUTER_ENCODERS.get(backend_key)
    if encoder_type is None:
        supported = ", ".join(sorted(ROUTER_ENCODERS))
        raise ValueError(
            f"Unsupported Mnemo router backend {backend!r}. Supported backends: {supported}."
        )
    encoder = encoder_type(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
    )
    return Router(
        encoder=encoder,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
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
        return Router._empty_decision(len(queries))
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
