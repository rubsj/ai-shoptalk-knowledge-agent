"""BM25 retriever — sparse lexical retrieval using rank-bm25.

BM25Okapi takes tokenized corpus (list of token lists). Scores are unbounded
[0, ∞) — MUST be normalized before combining with cosine scores in hybrid retrieval.

Used as both a standalone baseline (5 BM25-only configs in the experiment grid)
and as the sparse component inside HybridRetriever.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.interfaces import BaseRetriever
from src.schemas import Chunk, RetrievalResult


class BM25Retriever(BaseRetriever):
    """BM25Okapi lexical retrieval over a fixed corpus of chunks.

    Corpus is built at construction time. Not suitable for incremental updates —
    rebuild if the chunk set changes.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if not chunks:
            self._bm25: BM25Okapi | None = None
        else:
            tokenized = [c.content.lower().split() for c in chunks]
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Score all chunks with BM25, return top-k descending by score."""
        if not self._chunks or self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores: np.ndarray = self._bm25.get_scores(tokens)
        # argsort ascending → reverse for descending, take top_k
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                retriever_type="bm25",
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]
