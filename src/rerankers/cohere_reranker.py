"""Cohere reranker — uses Cohere's rerank API to re-score retrieved chunks.

Takes top-20 candidates from retrieval, calls Cohere rerank-english-v3.0,
returns top-k re-ordered by relevance score.

Why Cohere API: production reranking without local model memory overhead.
Limited to top-20 candidates to stay within Cohere free tier limits.
"""

from __future__ import annotations

import logging
import os
import time

import cohere

from src.interfaces import BaseReranker
from src.schemas import RetrievalResult

logger = logging.getLogger(__name__)

_MODEL = "rerank-english-v3.0"
_MAX_RETRIES = 5
_RETRY_DELAY_SECONDS = 10  # trial key = 10 calls/min, need >6s between calls


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

        # Retry with backoff for trial key rate limits (10 calls/min)
        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.rerank(
                    model=_MODEL,
                    query=query,
                    documents=documents,
                    top_n=min(top_k, len(results)),
                )
                break
            except cohere.errors.too_many_requests_error.TooManyRequestsError:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY_SECONDS * (attempt + 1)
                    logger.warning("Cohere rate limit hit, retrying in %ds (attempt %d/%d)",
                                   wait, attempt + 1, _MAX_RETRIES)
                    time.sleep(wait)
                else:
                    raise

        if response is None:
            return results[:top_k]  # fallback: return original order

        return [
            RetrievalResult(
                chunk=results[item.index].chunk,
                score=float(item.relevance_score),
                retriever_type=results[item.index].retriever_type,
                rank=i + 1,
            )
            for i, item in enumerate(response.results)
        ]
