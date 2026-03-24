"""Retriever implementations — all extend BaseRetriever.

Retrievers: DenseRetriever, BM25Retriever, HybridRetriever.
HybridRetriever uses internal composition (owns DenseRetriever + BM25Retriever).
"""

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever

__all__ = ["DenseRetriever", "BM25Retriever", "HybridRetriever"]
