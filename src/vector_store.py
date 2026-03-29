"""FAISS vector store with explicit persistence.

FAISSVectorStore implements BaseVectorStore using IndexFlatIP (inner product).
Vectors must be L2-normalized before add() and search() — IndexFlatIP equals
cosine similarity only on unit vectors.

Why FAISS over ChromaDB: P4 used ChromaDB. P5 needs low-level control for
35+ experiment configs — explicit index type selection, direct save/load,
no metadata overhead. Different tools for different jobs.

Persistence: two files per path prefix:
  <path>.faiss — raw FAISS binary index
  <path>.json  — chunk metadata (embedding excluded; lives in the FAISS binary)
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.interfaces import BaseVectorStore
from src.schemas import Chunk, ChunkMetadata


class FAISSVectorStore(BaseVectorStore):
    """Exact inner-product search over L2-normalised embeddings.

    Caller is responsible for passing L2-normalised embeddings via add().
    search() normalises the query internally so it is safe to pass raw query
    embeddings directly from the embedder.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # BaseVectorStore implementation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their pre-computed embeddings to the index.

        Args:
            chunks: Chunk objects parallel to the embeddings array.
            embeddings: shape (len(chunks), dimension), expected L2-normalised.

        Raises:
            ValueError: if shapes are inconsistent.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) != embeddings rows ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"embedding dimension {embeddings.shape[1]} != index dimension {self._dimension}"
            )
        # WHY: contiguous float32 copy before normalize_L2 avoids mutating caller's array
        vecs = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._chunks.extend(chunks)

    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Top-K similarity search.

        Returns list of (Chunk, score) sorted descending by score (cosine sim).
        Returns fewer than top_k items if the index has fewer vectors.
        """
        if self._index.ntotal == 0:
            return []
        # Reshape to (1, dim) and normalise — safe to mutate this copy
        query = np.ascontiguousarray(query_embedding.astype(np.float32).reshape(1, -1))
        faiss.normalize_L2(query)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        """Persist index and chunk metadata to <path>.faiss and <path>.json."""
        faiss.write_index(self._index, f"{path}.faiss")
        # Exclude embedding (np.ndarray) — not JSON-serialisable; it lives in FAISS binary
        chunks_data = [
            {
                "id": c.id,
                "content": c.content,
                "metadata": c.metadata.model_dump(),
            }
            for c in self._chunks
        ]
        Path(f"{path}.json").write_text(json.dumps(chunks_data, indent=2))

    def load(self, path: str) -> None:
        """Load index and chunk metadata from <path>.faiss and <path>.json."""
        self._index = faiss.read_index(f"{path}.faiss")
        raw = json.loads(Path(f"{path}.json").read_text())
        self._chunks = [
            Chunk(
                id=item["id"],
                content=item["content"],
                metadata=ChunkMetadata(**item["metadata"]),
            )
            for item in raw
        ]

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        """Read-only access to loaded chunks — needed by create_retriever() for BM25/hybrid."""
        return list(self._chunks)

    def __len__(self) -> int:
        return self._index.ntotal
