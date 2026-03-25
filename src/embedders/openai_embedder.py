"""OpenAI embedder — text-embedding-3-small, 1536-dimensional embeddings.

API-based embedder: zero local RAM, batched requests (100 texts/call).
Provides build-vs-buy comparison: local model quality vs managed API.

Why text-embedding-3-small over large: cost (5× cheaper) with competitive
quality for retrieval. Upgrade to text-embedding-3-large in extended experiments
if quality gap is significant.

Runs LAST in experiment grid — after all local model experiments complete.
"""

from __future__ import annotations

import faiss
import litellm
import numpy as np

from src.interfaces import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """1536-dimensional embedder using text-embedding-3-small via API.

    Zero local RAM footprint — all compute is remote. Batches at 100 texts/call
    to stay within OpenAI's input limits.
    """

    _MODEL_NAME = "text-embedding-3-small"
    _DIMENSIONS = 1536
    _BATCH_SIZE = 100

    def __init__(self) -> None:
        pass  # API-based — nothing to load locally

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed texts. Returns shape (len(texts), 1536), L2-normalised."""
        if not texts:
            return np.empty((0, self._DIMENSIONS), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = litellm.embedding(model=self._MODEL_NAME, input=batch)
            all_embeddings.extend(item["embedding"] for item in response.data)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embed. Returns shape (1536,), L2-normalised."""
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        """1536 — FAISS IndexFlatIP must be initialized with this value."""
        return self._DIMENSIONS
