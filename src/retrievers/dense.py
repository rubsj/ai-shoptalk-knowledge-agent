"""Dense retriever — cosine similarity search over FAISS index.

Embeds query with the same model used during ingestion, searches FAISS
IndexFlatIP, returns top-k RetrievalResults with cosine similarity scores.

Requires L2-normalized query vector (faiss.normalize_L2) to match index vectors.
"""

from __future__ import annotations

from src.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from src.schemas import RetrievalResult


class DenseRetriever(BaseRetriever):
    """Cosine similarity retrieval via FAISS IndexFlatIP.

    Delegates embedding to BaseEmbedder and search to BaseVectorStore so
    this class is model- and index-agnostic.
    """

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Embed query, search FAISS, return ranked RetrievalResults."""
        query_vec = self._embedder.embed_query(query)
        hits = self._vector_store.search(query_vec, top_k)
        return [
            RetrievalResult(chunk=chunk, score=score, retriever_type="dense", rank=i + 1)
            for i, (chunk, score) in enumerate(hits)
        ]
