"""mpnet embedder — all-mpnet-base-v2, 768-dimensional embeddings.

Larger model than MiniLM (110M params, ~420MB). Higher quality embeddings
at the cost of slower inference. Same dimensionality as nomic-embed-text
(Ollama Day 5) — enables direct quality comparison on the same FAISS structure.
"""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.interfaces import BaseEmbedder


class MpnetEmbedder(BaseEmbedder):
    """768-dimensional bi-encoder using all-mpnet-base-v2.

    Best local SentenceTransformer model for retrieval quality. Use when
    embedding quality matters more than ingestion speed.
    """

    _MODEL_NAME = "all-mpnet-base-v2"
    _DIMENSIONS = 768

    def __init__(self, device: str | None = None) -> None:
        # WHY: eager load — same rationale as MiniLMEmbedder (128GB, no pressure)
        # WHY device param: MPS (Apple Silicon GPU) crashes on large batches in some
        # sandbox environments. Pass device="cpu" as fallback.
        self._model = SentenceTransformer(self._MODEL_NAME, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed texts. Returns shape (len(texts), 768), L2-normalised."""
        if not texts:
            return np.empty((0, self._DIMENSIONS), dtype=np.float32)
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embed. Returns shape (768,), L2-normalised."""
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        """768 — FAISS IndexFlatIP must be initialized with this value."""
        return self._DIMENSIONS
