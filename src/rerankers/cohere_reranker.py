"""Cohere reranker — uses Cohere's rerank API to re-score retrieved chunks.

Takes top-20 candidates from retrieval, calls Cohere rerank-english-v3.0,
returns top-k re-ordered by relevance score.

Why Cohere API: production reranking without local model memory overhead.
Limited to top-20 candidates to stay within Cohere free tier limits.
"""

from __future__ import annotations

import os

import cohere

from src.interfaces import BaseReranker
from src.schemas import RetrievalResult

_MODEL = "rerank-english-v3.0"


class CohereReranker(BaseReranker):
    """Cohere API reranker using rerank-english-v3.0."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = cohere.ClientV2(api_key or os.environ["COHERE_API_KEY"])

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Call Cohere rerank API, map results back to original RetrievalResult objects."""
        if not results:
            return []

        documents = [r.chunk.content for r in results]
        response = self._client.rerank(
            model=_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_k, len(results)),
        )

        return [
            RetrievalResult(
                chunk=results[item.index].chunk,
                score=float(item.relevance_score),
                retriever_type=results[item.index].retriever_type,
                rank=i + 1,
            )
            for i, item in enumerate(response.results)
        ]
