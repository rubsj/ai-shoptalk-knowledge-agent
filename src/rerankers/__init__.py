"""Reranker implementations — all extend BaseReranker.

Rerankers: CohereReranker (API), CrossEncoderReranker (local sentence-transformers).
Both rerank top-20 candidates only — bi-encoder retrieves, cross-encoder refines.
"""

from src.rerankers.cohere_reranker import CohereReranker
from src.rerankers.cross_encoder import CrossEncoderReranker

__all__ = ["CrossEncoderReranker", "CohereReranker"]
