"""Cross-encoder reranker — local sentence-transformers CrossEncoder.

Takes top-20 candidates, scores each (query, chunk) pair through a
cross-encoder model (ms-marco-MiniLM-L-6-v2). More accurate than bi-encoder
because it attends to both query and document jointly.

Load → rerank → del → gc.collect() pattern to avoid holding model in memory
between reranking calls (memory budget constraint on M2 8GB).

Why cross-encoder after bi-encoder: bi-encoder retrieval is O(1) at query time
(pre-computed embeddings). Cross-encoder is O(n) — only feasible on top-20.
"""

from __future__ import annotations

import gc

from sentence_transformers import CrossEncoder

from src.interfaces import BaseReranker
from src.schemas import RetrievalResult

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Score (query, chunk) pairs, return top_k sorted by cross-encoder score."""
        if not results:
            return []

        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(results, scores), key=lambda x: float(x[1]), reverse=True
        )
        top = scored[:top_k]

        return [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
                retriever_type=result.retriever_type,
                rank=i + 1,
            )
            for i, (result, score) in enumerate(top)
        ]
