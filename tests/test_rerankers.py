"""Tests for CrossEncoderReranker and CohereReranker.

Both: mocked model/API — no network or GPU needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.interfaces import BaseReranker
from src.rerankers import CohereReranker, CrossEncoderReranker
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


def _make_result(
    content: str, score: float, retriever_type: str = "dense", rank: int = 1
) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(content),
        score=score,
        retriever_type=retriever_type,  # type: ignore[arg-type]
        rank=rank,
    )


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def _make_reranker(self, predict_scores: list[float]) -> CrossEncoderReranker:
        with patch("src.rerankers.cross_encoder.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            mock_model.predict.return_value = predict_scores
            MockCE.return_value = mock_model
            reranker = CrossEncoderReranker()
        reranker._model = mock_model
        return reranker

    def test_returns_retrieval_results(self) -> None:
        reranker = self._make_reranker([0.9, 0.5])
        results = [_make_result("a", 0.8, rank=1), _make_result("b", 0.6, rank=2)]
        out = reranker.rerank("query", results, top_k=2)
        assert all(isinstance(r, RetrievalResult) for r in out)

    def test_top_k_respected(self) -> None:
        reranker = self._make_reranker([0.9, 0.5, 0.3])
        results = [_make_result(f"doc{i}", 0.8 - i * 0.1, rank=i + 1) for i in range(3)]
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2

    def test_reranks_by_cross_encoder_score(self) -> None:
        """Lower-ranked input chunk gets higher cross-encoder score → moves to rank 1."""
        reranker = self._make_reranker([0.2, 0.9])  # doc1 scored higher by CE
        doc0 = _make_result("doc0", 0.9, rank=1)
        doc1 = _make_result("doc1", 0.5, rank=2)
        out = reranker.rerank("query", [doc0, doc1], top_k=2)
        assert out[0].chunk.content == "doc1"

    def test_ranks_sequential_from_one(self) -> None:
        reranker = self._make_reranker([0.9, 0.6, 0.3])
        results = [_make_result(f"d{i}", 0.8, rank=i + 1) for i in range(3)]
        out = reranker.rerank("query", results, top_k=3)
        assert [r.rank for r in out] == [1, 2, 3]

    def test_scores_replaced_by_cross_encoder(self) -> None:
        reranker = self._make_reranker([0.77])
        results = [_make_result("doc", 0.5, rank=1)]
        out = reranker.rerank("query", results, top_k=1)
        assert abs(out[0].score - 0.77) < 1e-6

    def test_retriever_type_preserved(self) -> None:
        reranker = self._make_reranker([0.8])
        results = [_make_result("doc", 0.5, retriever_type="hybrid", rank=1)]
        out = reranker.rerank("query", results, top_k=1)
        assert out[0].retriever_type == "hybrid"

    def test_empty_results_returns_empty(self) -> None:
        reranker = self._make_reranker([])
        assert reranker.rerank("query", [], top_k=5) == []

    def test_predict_called_with_pairs(self) -> None:
        reranker = self._make_reranker([0.5, 0.3])
        results = [_make_result("alpha text", 0.8, rank=1), _make_result("beta text", 0.6, rank=2)]
        reranker.rerank("what is alpha?", results, top_k=2)
        call_args = reranker._model.predict.call_args[0][0]
        assert call_args == [("what is alpha?", "alpha text"), ("what is alpha?", "beta text")]

    def test_implements_base_reranker(self) -> None:
        with patch("src.rerankers.cross_encoder.CrossEncoder"):
            reranker = CrossEncoderReranker()
        assert isinstance(reranker, BaseReranker)


# ---------------------------------------------------------------------------
# CohereReranker
# ---------------------------------------------------------------------------


def _make_cohere_item(index: int, relevance_score: float) -> MagicMock:
    item = MagicMock()
    item.index = index
    item.relevance_score = relevance_score
    return item


class TestCohereReranker:
    def _make_reranker(self, rerank_items: list[MagicMock]) -> CohereReranker:
        with patch("src.rerankers.cohere_reranker.cohere.ClientV2") as MockClient:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.results = rerank_items
            mock_client.rerank.return_value = mock_response
            MockClient.return_value = mock_client
            reranker = CohereReranker(api_key="test-key")
        reranker._client = mock_client
        return reranker

    def test_returns_retrieval_results(self) -> None:
        items = [_make_cohere_item(0, 0.95), _make_cohere_item(1, 0.7)]
        reranker = self._make_reranker(items)
        results = [_make_result("a", 0.8, rank=1), _make_result("b", 0.6, rank=2)]
        out = reranker.rerank("query", results, top_k=2)
        assert all(isinstance(r, RetrievalResult) for r in out)

    def test_top_k_passed_to_api(self) -> None:
        items = [_make_cohere_item(0, 0.9)]
        reranker = self._make_reranker(items)
        results = [_make_result(f"doc{i}", 0.8, rank=i + 1) for i in range(5)]
        reranker.rerank("query", results, top_k=3)
        call_kwargs = reranker._client.rerank.call_args[1]
        assert call_kwargs.get("top_n") == 3

    def test_reranks_by_cohere_score(self) -> None:
        """Cohere returns doc1 first (higher relevance) regardless of original order."""
        items = [_make_cohere_item(1, 0.95), _make_cohere_item(0, 0.4)]
        reranker = self._make_reranker(items)
        doc0 = _make_result("original first", 0.9, rank=1)
        doc1 = _make_result("original second", 0.5, rank=2)
        out = reranker.rerank("query", [doc0, doc1], top_k=2)
        assert out[0].chunk.content == "original second"

    def test_ranks_sequential_from_one(self) -> None:
        items = [_make_cohere_item(2, 0.9), _make_cohere_item(0, 0.7), _make_cohere_item(1, 0.5)]
        reranker = self._make_reranker(items)
        results = [_make_result(f"d{i}", 0.8, rank=i + 1) for i in range(3)]
        out = reranker.rerank("query", results, top_k=3)
        assert [r.rank for r in out] == [1, 2, 3]

    def test_scores_replaced_by_cohere_score(self) -> None:
        items = [_make_cohere_item(0, 0.88)]
        reranker = self._make_reranker(items)
        results = [_make_result("doc", 0.5, rank=1)]
        out = reranker.rerank("query", results, top_k=1)
        assert abs(out[0].score - 0.88) < 1e-6

    def test_retriever_type_preserved(self) -> None:
        items = [_make_cohere_item(0, 0.8)]
        reranker = self._make_reranker(items)
        results = [_make_result("doc", 0.5, retriever_type="bm25", rank=1)]
        out = reranker.rerank("query", results, top_k=1)
        assert out[0].retriever_type == "bm25"

    def test_empty_results_returns_empty(self) -> None:
        reranker = self._make_reranker([])
        assert reranker.rerank("query", [], top_k=5) == []

    def test_api_called_with_correct_documents(self) -> None:
        items = [_make_cohere_item(0, 0.9), _make_cohere_item(1, 0.5)]
        reranker = self._make_reranker(items)
        results = [_make_result("first doc", 0.8, rank=1), _make_result("second doc", 0.6, rank=2)]
        reranker.rerank("find something", results, top_k=2)
        call_kwargs = reranker._client.rerank.call_args[1]
        assert call_kwargs.get("documents") == ["first doc", "second doc"]
        assert call_kwargs.get("query") == "find something"

    def test_top_n_capped_at_results_length(self) -> None:
        """top_k > len(results) → top_n capped to len(results)."""
        items = [_make_cohere_item(0, 0.9)]
        reranker = self._make_reranker(items)
        results = [_make_result("only one", 0.8, rank=1)]
        reranker.rerank("query", results, top_k=10)
        call_kwargs = reranker._client.rerank.call_args[1]
        assert call_kwargs.get("top_n") == 1

    def test_implements_base_reranker(self) -> None:
        with patch("src.rerankers.cohere_reranker.cohere.ClientV2"):
            reranker = CohereReranker(api_key="test")
        assert isinstance(reranker, BaseReranker)
