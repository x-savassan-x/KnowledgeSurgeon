from abc import ABC, abstractmethod
from dataclasses import dataclass

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

    def encode_queries(self, prompts: list[str]) -> torch.Tensor:
        return self.encode(prompts)

    def encode_keys(self, prompts: list[str]) -> torch.Tensor:
        return self.encode(prompts)


class SentenceTransformerEncoder(PromptEncoder):
    """Dense encoder for the SentenceTransformers model family.

    E5 models require asymmetric ``query:`` and ``passage:`` prefixes; other
    registered models use the prompt text unchanged.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        use_fp16: bool = False,
        max_length: int = 384,
        batch_size: int = 64,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        embedder=None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = str(device)
        self.use_fp16 = bool(use_fp16 and self.device.startswith("cuda"))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        is_e5 = "e5" in self.model_name.casefold()
        self.query_prefix = (
            "query: " if query_prefix is None and is_e5 else str(query_prefix or "")
        )
        self.passage_prefix = (
            "passage: " if passage_prefix is None and is_e5 else str(passage_prefix or "")
        )
        self._embedder = embedder

    def load(self) -> None:
        self._get_embedder()

    def encode(self, prompts: list[str]) -> torch.Tensor:
        return self._encode(prompts, role="query")

    def encode_queries(self, prompts: list[str]) -> torch.Tensor:
        return self._encode(prompts, role="query")

    def encode_keys(self, prompts: list[str]) -> torch.Tensor:
        return self._encode(prompts, role="passage")

    def _encode(self, prompts: list[str], *, role: str) -> torch.Tensor:
        if not prompts:
            return torch.empty((0, 0), dtype=torch.float32, device="cpu")
        if role not in {"query", "passage", "symmetric"}:
            raise ValueError("SentenceTransformer prompt role must be query, passage, or symmetric")
        prefix = self._prefix_for(role)
        formatted = [f"{prefix}{prompt}" for prompt in prompts]
        model = self._get_embedder()
        encoded = model.encode(
            formatted,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = _cpu_float_matrix(encoded)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"Unexpected embedding shape {tuple(embeddings.shape)} for {len(prompts)} prompts"
            )
        return embeddings

    def _prefix_for(self, role: str) -> str:
        return {
            "query": self.query_prefix,
            "passage": self.passage_prefix,
            "symmetric": self.query_prefix,
        }[role]

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name, device=self.device)
            self._embedder.max_seq_length = min(
                int(self._embedder.max_seq_length), self.max_length
            )
            if self.use_fp16:
                self._embedder.half()
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

        encode = getattr(self.encoder, "encode_queries", self.encoder.encode)
        embeddings = _normalize_rows(_cpu_float_matrix(encode(prompts)))
        if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"Unexpected embedding shape {tuple(embeddings.shape)} for {len(prompts)} prompts"
            )
        return embeddings

    def encode_keys(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            dimension = self.embedding_dimension or 0
            return torch.empty((0, dimension), dtype=torch.float32, device="cpu")

        encode = getattr(self.encoder, "encode_keys", self.encoder.encode)
        embeddings = _normalize_rows(_cpu_float_matrix(encode(prompts)))
        if embeddings.ndim != 2 or embeddings.shape[0] != len(prompts):
            raise ValueError(
                f"Unexpected embedding shape {tuple(embeddings.shape)} for {len(prompts)} prompts"
            )
        return embeddings

    def add_prompts(self, prompts: list[str]) -> torch.Tensor:
        embeddings = self.encode_keys(prompts)
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
    "bge-base-en-v1.5": SentenceTransformerEncoder,
    "e5-base-v2": SentenceTransformerEncoder,
    "all-minilm-l6-v2": SentenceTransformerEncoder,
}

ROUTER_PREFIXES: dict[str, tuple[str, str]] = {
    "bge-base-en-v1.5": ("", ""),
    "e5-base-v2": ("query: ", "passage: "),
    "all-minilm-l6-v2": ("", ""),
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
    query_prefix, passage_prefix = ROUTER_PREFIXES[backend_key]
    encoder = encoder_type(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        max_length=max_length,
        batch_size=batch_size,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
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
