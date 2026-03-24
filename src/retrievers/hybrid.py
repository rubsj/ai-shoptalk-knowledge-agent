"""Hybrid retriever — weighted combination of dense and BM25 retrieval.

Internal composition: owns both DenseRetriever and BM25Retriever (PRD Decision 2).
Score fusion:
  1. Retrieve top-k from dense (cosine [0,1])
  2. Retrieve top-k from BM25 (unbounded [0,∞))
  3. Min-max normalize BM25 scores to [0,1]
  4. Merge result sets: combined_score = α * dense_score + (1-α) * bm25_score
  5. Re-rank by combined score, return top-k

Why min-max normalize: naive combination without normalization lets BM25 dominate
because its scores can be 10-100× larger than cosine scores in [0,1].

Config param: alpha (default 0.7) — higher α = more weight on dense retrieval.
"""

from __future__ import annotations

from src.interfaces import BaseRetriever
from src.schemas import RetrievalResult


class HybridRetriever(BaseRetriever):
    """Weighted score fusion: alpha * dense + (1-alpha) * bm25_normalized."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        alpha: float = 0.7,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._alpha = alpha

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Fuse dense and BM25 scores, return top-k by combined score."""
        oversample = top_k * 2

        dense_results = self._dense.retrieve(query, oversample)
        bm25_results = self._bm25.retrieve(query, oversample)

        # Build score maps keyed by chunk ID
        dense_scores: dict[str, tuple[RetrievalResult, float]] = {
            r.chunk.id: (r, r.score) for r in dense_results
        }
        bm25_scores: dict[str, float] = {r.chunk.id: r.score for r in bm25_results}

        # Min-max normalize BM25 scores to [0, 1]
        if bm25_scores:
            raw_vals = list(bm25_scores.values())
            bm25_min, bm25_max = min(raw_vals), max(raw_vals)
            if bm25_max == bm25_min:
                # All identical scores — set to 0.0 (no discriminative signal)
                bm25_norm: dict[str, float] = {k: 0.0 for k in bm25_scores}
            else:
                bm25_norm = {
                    k: (v - bm25_min) / (bm25_max - bm25_min)
                    for k, v in bm25_scores.items()
                }
        else:
            bm25_norm = {}

        # Union of chunk IDs from both retrievers
        all_ids = set(dense_scores) | set(bm25_norm)

        # Resolve chunks and combined scores
        chunks_by_id: dict[str, object] = {}
        for cid, (result, _) in dense_scores.items():
            chunks_by_id[cid] = result.chunk
        for result in bm25_results:
            if result.chunk.id not in chunks_by_id:
                chunks_by_id[result.chunk.id] = result.chunk

        combined: list[tuple[object, float]] = []
        for cid in all_ids:
            d_score = dense_scores[cid][1] if cid in dense_scores else 0.0
            b_score = bm25_norm.get(cid, 0.0)
            score = self._alpha * d_score + (1.0 - self._alpha) * b_score
            combined.append((chunks_by_id[cid], score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top = combined[:top_k]

        return [
            RetrievalResult(chunk=chunk, score=score, retriever_type="hybrid", rank=i + 1)  # type: ignore[arg-type]
            for i, (chunk, score) in enumerate(top)
        ]
