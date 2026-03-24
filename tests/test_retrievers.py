"""Tests for DenseRetriever, BM25Retriever, HybridRetriever.

Dense: mocked embedder + vector store (no real models needed).
BM25: real rank_bm25 (lightweight, no network).
Hybrid: mocked sub-retrievers with known scores — focus on normalization logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.interfaces import BaseRetriever
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.schemas import Chunk, ChunkMetadata, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "hello world", doc_id: str = "doc1") -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            source="test.pdf",
            page_number=0,
            start_char=0,
            end_char=len(content),
            chunk_index=0,
        ),
    )


def _make_retrieval_result(
    chunk: Chunk, score: float, retriever_type: str = "dense", rank: int = 1
) -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, retriever_type=retriever_type, rank=rank)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    def _make_store(self, chunks_and_scores: list[tuple[Chunk, float]]) -> MagicMock:
        store = MagicMock()
        store.search.return_value = chunks_and_scores
        return store

    def _make_embedder(self, dim: int = 8) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_query.return_value = np.random.randn(dim).astype(np.float32)
        return embedder

    def test_returns_retrieval_results(self) -> None:
        chunk = _make_chunk("some text")
        store = self._make_store([(chunk, 0.95)])
        embedder = self._make_embedder()
        retriever = DenseRetriever(embedder, store)

        results = retriever.retrieve("query", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)

    def test_retriever_type_is_dense(self) -> None:
        chunk = _make_chunk()
        store = self._make_store([(chunk, 0.8)])
        retriever = DenseRetriever(self._make_embedder(), store)

        results = retriever.retrieve("q", top_k=1)
        assert results[0].retriever_type == "dense"

    def test_ranks_are_sequential_from_one(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        store = self._make_store([(c, 0.9 - i * 0.1) for i, c in enumerate(chunks)])
        retriever = DenseRetriever(self._make_embedder(), store)

        results = retriever.retrieve("q", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_preserved(self) -> None:
        chunk = _make_chunk()
        store = self._make_store([(chunk, 0.77)])
        retriever = DenseRetriever(self._make_embedder(), store)

        results = retriever.retrieve("q", top_k=1)
        assert abs(results[0].score - 0.77) < 1e-6

    def test_embed_query_called_with_query(self) -> None:
        chunk = _make_chunk()
        store = self._make_store([(chunk, 0.5)])
        embedder = self._make_embedder()
        retriever = DenseRetriever(embedder, store)

        retriever.retrieve("what is attention?", top_k=1)
        embedder.embed_query.assert_called_once_with("what is attention?")

    def test_store_searched_with_top_k(self) -> None:
        store = self._make_store([])
        embedder = self._make_embedder()
        retriever = DenseRetriever(embedder, store)

        retriever.retrieve("q", top_k=7)
        args, kwargs = store.search.call_args
        # args[1] is top_k (positional); kwargs may have it as keyword
        assert args[1] == 7 or kwargs.get("top_k") == 7

    def test_empty_store_returns_empty(self) -> None:
        store = self._make_store([])
        retriever = DenseRetriever(self._make_embedder(), store)
        assert retriever.retrieve("q", top_k=5) == []

    def test_implements_base_retriever(self) -> None:
        store = self._make_store([])
        retriever = DenseRetriever(self._make_embedder(), store)
        assert isinstance(retriever, BaseRetriever)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    def test_retrieve_returns_top_k(self) -> None:
        chunks = [_make_chunk(f"word{i} extra text") for i in range(5)]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("word1 extra", top_k=3)
        assert len(results) == 3

    def test_retriever_type_is_bm25(self) -> None:
        chunks = [_make_chunk("hello world"), _make_chunk("foo bar")]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("hello", top_k=1)
        assert results[0].retriever_type == "bm25"

    def test_ranks_sequential_from_one(self) -> None:
        chunks = [_make_chunk(f"token{i}") for i in range(4)]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("token0", top_k=4)
        assert [r.rank for r in results] == [1, 2, 3, 4]

    def test_scores_descending(self) -> None:
        chunks = [_make_chunk(f"item{i}") for i in range(5)]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("item0", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_chunk_ranks_first(self) -> None:
        chunks = [
            _make_chunk("python programming language"),
            _make_chunk("chocolate ice cream dessert"),
            _make_chunk("machine learning python"),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("python machine learning", top_k=3)
        top_content = results[0].chunk.content
        assert "python" in top_content or "machine" in top_content

    def test_empty_corpus_returns_empty(self) -> None:
        retriever = BM25Retriever([])
        assert retriever.retrieve("any query", top_k=5) == []

    def test_top_k_larger_than_corpus_returns_all(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("c0", top_k=100)
        assert len(results) == 3

    def test_implements_base_retriever(self) -> None:
        retriever = BM25Retriever([_make_chunk()])
        assert isinstance(retriever, BaseRetriever)

    def test_score_is_float(self) -> None:
        chunks = [_make_chunk("hello world")]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("hello", top_k=1)
        assert isinstance(results[0].score, float)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


def _mock_retriever(results: list[RetrievalResult]) -> MagicMock:
    mock = MagicMock(spec=BaseRetriever)
    mock.retrieve.return_value = results
    return mock


class TestHybridRetriever:
    def _two_chunk_setup(self) -> tuple[Chunk, Chunk]:
        c1 = _make_chunk("chunk one content")
        c2 = _make_chunk("chunk two content")
        return c1, c2

    def test_returns_top_k_results(self) -> None:
        c1, c2 = self._two_chunk_setup()
        dense = _mock_retriever([
            _make_retrieval_result(c1, 0.9, "dense", 1),
            _make_retrieval_result(c2, 0.7, "dense", 2),
        ])
        bm25 = _mock_retriever([
            _make_retrieval_result(c1, 10.0, "bm25", 1),
            _make_retrieval_result(c2, 5.0, "bm25", 2),
        ])
        retriever = HybridRetriever(dense, bm25, alpha=0.7)
        results = retriever.retrieve("q", top_k=1)
        assert len(results) == 1

    def test_retriever_type_is_hybrid(self) -> None:
        c1, c2 = self._two_chunk_setup()
        dense = _mock_retriever([_make_retrieval_result(c1, 0.9, "dense", 1)])
        bm25 = _mock_retriever([_make_retrieval_result(c1, 5.0, "bm25", 1)])
        retriever = HybridRetriever(dense, bm25)
        results = retriever.retrieve("q", top_k=1)
        assert results[0].retriever_type == "hybrid"

    def test_ranks_sequential_from_one(self) -> None:
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        dense = _mock_retriever([_make_retrieval_result(c, 0.9 - i * 0.1, "dense", i + 1) for i, c in enumerate(chunks)])
        bm25 = _mock_retriever([_make_retrieval_result(c, 10.0 - i, "bm25", i + 1) for i, c in enumerate(chunks)])
        retriever = HybridRetriever(dense, bm25)
        results = retriever.retrieve("q", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_alpha_one_pure_dense(self) -> None:
        """alpha=1.0 means combined = dense score; BM25 has no influence."""
        c1, c2 = self._two_chunk_setup()
        # dense: c1 > c2; bm25: c2 > c1 (inverted preference)
        dense = _mock_retriever([
            _make_retrieval_result(c1, 0.9, "dense", 1),
            _make_retrieval_result(c2, 0.1, "dense", 2),
        ])
        bm25 = _mock_retriever([
            _make_retrieval_result(c2, 100.0, "bm25", 1),
            _make_retrieval_result(c1, 1.0, "bm25", 2),
        ])
        retriever = HybridRetriever(dense, bm25, alpha=1.0)
        results = retriever.retrieve("q", top_k=2)
        assert results[0].chunk.id == c1.id

    def test_alpha_zero_pure_bm25(self) -> None:
        """alpha=0.0 means combined = bm25_norm; dense has no influence."""
        c1, c2 = self._two_chunk_setup()
        # dense: c1 > c2; bm25: c2 >> c1
        dense = _mock_retriever([
            _make_retrieval_result(c1, 0.9, "dense", 1),
            _make_retrieval_result(c2, 0.1, "dense", 2),
        ])
        bm25 = _mock_retriever([
            _make_retrieval_result(c2, 20.0, "bm25", 1),
            _make_retrieval_result(c1, 1.0, "bm25", 2),
        ])
        retriever = HybridRetriever(dense, bm25, alpha=0.0)
        results = retriever.retrieve("q", top_k=2)
        assert results[0].chunk.id == c2.id

    def test_identical_bm25_scores_edge_case(self) -> None:
        """All BM25 scores identical → normalized to 0.0; only dense drives order."""
        c1, c2 = self._two_chunk_setup()
        dense = _mock_retriever([
            _make_retrieval_result(c1, 0.9, "dense", 1),
            _make_retrieval_result(c2, 0.4, "dense", 2),
        ])
        bm25 = _mock_retriever([
            _make_retrieval_result(c1, 5.0, "bm25", 1),
            _make_retrieval_result(c2, 5.0, "bm25", 2),
        ])
        retriever = HybridRetriever(dense, bm25, alpha=0.7)
        results = retriever.retrieve("q", top_k=2)
        # Dense dominates since BM25 normalized to 0 for both
        assert results[0].chunk.id == c1.id

    def test_chunk_only_in_bm25_included(self) -> None:
        """A chunk returned only by BM25 (not in dense results) should still appear."""
        c1 = _make_chunk("dense only")
        c2 = _make_chunk("bm25 only")
        dense = _mock_retriever([_make_retrieval_result(c1, 0.8, "dense", 1)])
        bm25 = _mock_retriever([_make_retrieval_result(c2, 10.0, "bm25", 1)])
        retriever = HybridRetriever(dense, bm25, alpha=0.5)
        results = retriever.retrieve("q", top_k=2)
        result_ids = {r.chunk.id for r in results}
        assert c2.id in result_ids

    def test_chunk_only_in_dense_included(self) -> None:
        """A chunk returned only by dense (not in BM25) should still appear."""
        c1 = _make_chunk("dense only")
        c2 = _make_chunk("bm25 only")
        dense = _mock_retriever([_make_retrieval_result(c1, 0.8, "dense", 1)])
        bm25 = _mock_retriever([_make_retrieval_result(c2, 10.0, "bm25", 1)])
        retriever = HybridRetriever(dense, bm25, alpha=0.5)
        results = retriever.retrieve("q", top_k=2)
        result_ids = {r.chunk.id for r in results}
        assert c1.id in result_ids

    def test_empty_dense_and_bm25_returns_empty(self) -> None:
        dense = _mock_retriever([])
        bm25 = _mock_retriever([])
        retriever = HybridRetriever(dense, bm25)
        assert retriever.retrieve("q", top_k=5) == []

    def test_invalid_alpha_raises(self) -> None:
        dense = _mock_retriever([])
        bm25 = _mock_retriever([])
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(dense, bm25, alpha=1.5)

    def test_implements_base_retriever(self) -> None:
        dense = _mock_retriever([])
        bm25 = _mock_retriever([])
        retriever = HybridRetriever(dense, bm25)
        assert isinstance(retriever, BaseRetriever)

    def test_oversample_passed_to_sub_retrievers(self) -> None:
        """HybridRetriever requests top_k*2 from each sub-retriever."""
        dense = _mock_retriever([])
        bm25 = _mock_retriever([])
        retriever = HybridRetriever(dense, bm25)
        retriever.retrieve("q", top_k=5)
        args_dense, _ = dense.retrieve.call_args
        args_bm25, _ = bm25.retrieve.call_args
        assert args_dense[1] == 10 or dense.retrieve.call_args[1].get("top_k") == 10
        assert args_bm25[1] == 10 or bm25.retrieve.call_args[1].get("top_k") == 10
