"""MiniLM embedder — all-MiniLM-L6-v2, 384-dimensional embeddings.

Smallest and fastest local model (22.7M params, ~90MB). Used for:
  - Dense retrieval experiments requiring 384d vectors
  - Embedding-semantic chunker boundary detection (always MiniLM, per PRD Decision 1)

Vectors are L2-normalized by SentenceTransformers by default. Verify with
np.linalg.norm(emb) ≈ 1.0 before FAISS IndexFlatIP add().
"""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.interfaces import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):
    """384-dimensional bi-encoder using all-MiniLM-L6-v2.

    Fastest local model. Use for high-throughput retrieval experiments
    where embedding quality is secondary to speed.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"
    _DIMENSIONS = 384

    def __init__(self) -> None:
        # WHY: eager load — 128GB M5 Max has no RAM pressure, lazy loading
        # just adds complexity for zero benefit here
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed texts. Returns shape (len(texts), 384), L2-normalised."""
        if not texts:
            return np.empty((0, self._DIMENSIONS), dtype=np.float32)
        embeddings = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        # WHY: copy() before normalize_L2 to avoid mutating the array in-place
        # if the caller holds a reference to the same buffer
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embed. Returns shape (384,), L2-normalised."""
        # WHY: reuse embed() so normalization path is identical for docs and queries
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        """384 — FAISS IndexFlatIP must be initialized with this value."""
        return self._DIMENSIONS
