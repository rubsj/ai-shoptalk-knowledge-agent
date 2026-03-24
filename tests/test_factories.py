"""Tests for factory functions in src/factories.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chunkers import (
    EmbeddingSemanticChunker,
    FixedSizeChunker,
    HeadingSemanticChunker,
    RecursiveChunker,
    SlidingWindowChunker,
)
from src.factories import (
    create_chunker,
    create_embedder,
    create_llm,
    create_reranker,
    create_retriever,
)
from src.interfaces import BaseChunker, BaseEmbedder, BaseLLM, BaseReranker, BaseRetriever
from src.schemas import Chunk, ChunkMetadata, ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25_config(**overrides) -> ExperimentConfig:
    defaults = dict(chunking_strategy="fixed", retriever_type="bm25")
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _dense_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        chunking_strategy="fixed",
        retriever_type="dense",
        embedding_model="minilm",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _hybrid_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        chunking_strategy="fixed",
        retriever_type="hybrid",
        embedding_model="minilm",
        hybrid_alpha=0.7,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_chunk(content: str = "hello", idx: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="doc1",
            source="test.pdf",
            page_number=idx,
            start_char=0,
            end_char=len(content),
            chunk_index=idx,
        ),
    )


# ---------------------------------------------------------------------------
# create_chunker
# ---------------------------------------------------------------------------


class TestCreateChunker:
    def test_fixed_returns_fixed_size_chunker(self) -> None:
        config = _bm25_config(chunking_strategy="fixed")
        chunker = create_chunker(config)
        assert isinstance(chunker, FixedSizeChunker)

    def test_fixed_passes_chunk_size(self) -> None:
        config = _bm25_config(chunking_strategy="fixed", chunk_size=256, chunk_overlap=25)
        chunker = create_chunker(config)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25

    def test_recursive_returns_recursive_chunker(self) -> None:
        config = _bm25_config(chunking_strategy="recursive")
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveChunker)

    def test_recursive_passes_chunk_size(self) -> None:
        config = _bm25_config(chunking_strategy="recursive", chunk_size=300, chunk_overlap=30)
        chunker = create_chunker(config)
        assert chunker.chunk_size == 300

    def test_sliding_window_returns_sliding_window_chunker(self) -> None:
        config = _bm25_config(
            chunking_strategy="sliding_window",
            window_size_tokens=100,
            step_size_tokens=50,
        )
        chunker = create_chunker(config)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_sliding_window_passes_window_and_step(self) -> None:
        config = _bm25_config(
            chunking_strategy="sliding_window",
            window_size_tokens=150,
            step_size_tokens=75,
        )
        chunker = create_chunker(config)
        assert chunker.window_size == 150
        assert chunker.step_size == 75

    def test_heading_semantic_returns_heading_chunker(self) -> None:
        config = _bm25_config(chunking_strategy="heading_semantic")
        chunker = create_chunker(config)
        assert isinstance(chunker, HeadingSemanticChunker)

    def test_heading_semantic_passes_min_chunk_size(self) -> None:
        config = _bm25_config(chunking_strategy="heading_semantic", min_chunk_size=200)
        chunker = create_chunker(config)
        assert chunker.min_chunk_size == 200

    def test_embedding_semantic_returns_embedding_chunker(self) -> None:
        config = _bm25_config(chunking_strategy="embedding_semantic")
        chunker = create_chunker(config)
        assert isinstance(chunker, EmbeddingSemanticChunker)

    def test_embedding_semantic_passes_threshold_and_min_size(self) -> None:
        config = _bm25_config(
            chunking_strategy="embedding_semantic",
            breakpoint_threshold=0.75,
            min_chunk_size=150,
        )
        chunker = create_chunker(config)
        assert chunker.breakpoint_threshold == 0.75
        assert chunker.min_chunk_size == 150

    def test_all_return_base_chunker(self) -> None:
        configs = [
            _bm25_config(chunking_strategy="fixed"),
            _bm25_config(chunking_strategy="recursive"),
            _bm25_config(
                chunking_strategy="sliding_window",
                window_size_tokens=100,
                step_size_tokens=50,
            ),
            _bm25_config(chunking_strategy="heading_semantic"),
            _bm25_config(chunking_strategy="embedding_semantic"),
        ]
        for config in configs:
            assert isinstance(create_chunker(config), BaseChunker)

    def test_unknown_strategy_raises(self) -> None:
        config = _bm25_config(chunking_strategy="fixed")
        config.__dict__["chunking_strategy"] = "unknown_strategy"
        with pytest.raises(ValueError, match="Unknown chunking_strategy"):
            create_chunker(config)


# ---------------------------------------------------------------------------
# create_embedder
# ---------------------------------------------------------------------------


class TestCreateEmbedder:
    def test_minilm_returns_minilm_embedder(self) -> None:
        with patch("src.embedders.minilm.SentenceTransformer"):
            from src.embedders import MiniLMEmbedder
            embedder = create_embedder("minilm")
        assert isinstance(embedder, MiniLMEmbedder)

    def test_mpnet_returns_mpnet_embedder(self) -> None:
        with patch("src.embedders.mpnet.SentenceTransformer"):
            from src.embedders import MpnetEmbedder
            embedder = create_embedder("mpnet")
        assert isinstance(embedder, MpnetEmbedder)

    def test_both_return_base_embedder(self) -> None:
        with patch("src.embedders.minilm.SentenceTransformer"), \
             patch("src.embedders.mpnet.SentenceTransformer"):
            assert isinstance(create_embedder("minilm"), BaseEmbedder)
            assert isinstance(create_embedder("mpnet"), BaseEmbedder)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            create_embedder("bert-large")


# ---------------------------------------------------------------------------
# create_retriever
# ---------------------------------------------------------------------------


class TestCreateRetriever:
    def test_bm25_returns_bm25_retriever(self) -> None:
        from src.retrievers import BM25Retriever
        config = _bm25_config()
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=None, chunks=chunks, vector_store=None)
        assert isinstance(retriever, BM25Retriever)

    def test_dense_returns_dense_retriever(self) -> None:
        from src.retrievers import DenseRetriever
        config = _dense_config()
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_store = MagicMock()
        retriever = create_retriever(config, embedder=mock_embedder, chunks=[], vector_store=mock_store)
        assert isinstance(retriever, DenseRetriever)

    def test_hybrid_returns_hybrid_retriever(self) -> None:
        from src.retrievers import HybridRetriever
        config = _hybrid_config(hybrid_alpha=0.6)
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_store = MagicMock()
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=mock_embedder, chunks=chunks, vector_store=mock_store)
        assert isinstance(retriever, HybridRetriever)

    def test_hybrid_alpha_passed_through(self) -> None:
        from src.retrievers import HybridRetriever
        config = _hybrid_config(hybrid_alpha=0.3)
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_store = MagicMock()
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=mock_embedder, chunks=chunks, vector_store=mock_store)
        assert isinstance(retriever, HybridRetriever)
        assert abs(retriever._alpha - 0.3) < 1e-9

    def test_hybrid_defaults_alpha_to_0_7_when_none(self) -> None:
        """hybrid_alpha=None on config → factory defaults to 0.7."""
        from src.retrievers import HybridRetriever
        config = _hybrid_config()
        # bypass pydantic to simulate None (normally validators enforce non-None for hybrid)
        config.__dict__["hybrid_alpha"] = None
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_store = MagicMock()
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=mock_embedder, chunks=chunks, vector_store=mock_store)
        assert abs(retriever._alpha - 0.7) < 1e-9

    def test_bm25_returns_base_retriever(self) -> None:
        config = _bm25_config()
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=None, chunks=chunks, vector_store=None)
        assert isinstance(retriever, BaseRetriever)

    def test_dense_returns_base_retriever(self) -> None:
        config = _dense_config()
        mock_embedder = MagicMock(spec=BaseEmbedder)
        retriever = create_retriever(config, embedder=mock_embedder, chunks=[], vector_store=MagicMock())
        assert isinstance(retriever, BaseRetriever)

    def test_hybrid_returns_base_retriever(self) -> None:
        config = _hybrid_config()
        mock_embedder = MagicMock(spec=BaseEmbedder)
        chunks = [_make_chunk()]
        retriever = create_retriever(config, embedder=mock_embedder, chunks=chunks, vector_store=MagicMock())
        assert isinstance(retriever, BaseRetriever)

    def test_unknown_retriever_raises(self) -> None:
        config = _bm25_config()
        config.__dict__["retriever_type"] = "unknown"
        with pytest.raises(ValueError, match="Unknown retriever_type"):
            create_retriever(config, embedder=None, chunks=[], vector_store=None)


# ---------------------------------------------------------------------------
# create_reranker
# ---------------------------------------------------------------------------


class TestCreateReranker:
    def test_cross_encoder_returns_cross_encoder_reranker(self) -> None:
        with patch("src.rerankers.cross_encoder.CrossEncoder"):
            from src.rerankers import CrossEncoderReranker
            reranker = create_reranker("cross_encoder")
        assert isinstance(reranker, CrossEncoderReranker)

    def test_cohere_returns_cohere_reranker(self) -> None:
        with patch("src.rerankers.cohere_reranker.cohere.ClientV2"), \
             patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            from src.rerankers import CohereReranker
            reranker = create_reranker("cohere")
        assert isinstance(reranker, CohereReranker)

    def test_cross_encoder_returns_base_reranker(self) -> None:
        with patch("src.rerankers.cross_encoder.CrossEncoder"):
            assert isinstance(create_reranker("cross_encoder"), BaseReranker)

    def test_cohere_returns_base_reranker(self) -> None:
        with patch("src.rerankers.cohere_reranker.cohere.ClientV2"), \
             patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            assert isinstance(create_reranker("cohere"), BaseReranker)

    def test_unknown_reranker_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reranker_type"):
            create_reranker("unknown_reranker")


# ---------------------------------------------------------------------------
# create_llm
# ---------------------------------------------------------------------------


class TestCreateLlm:
    def test_returns_litellm_client(self) -> None:
        from src.generator import LiteLLMClient
        llm = create_llm()
        assert isinstance(llm, LiteLLMClient)

    def test_returns_base_llm(self) -> None:
        llm = create_llm()
        assert isinstance(llm, BaseLLM)

    def test_model_forwarded(self) -> None:
        from src.generator import LiteLLMClient
        llm = create_llm(model="gpt-4o")
        assert isinstance(llm, LiteLLMClient)
        assert llm._model == "gpt-4o"

    def test_default_model_is_gpt_4o_mini(self) -> None:
        from src.generator import LiteLLMClient
        llm = create_llm()
        assert llm._model == "gpt-4o-mini"

    def test_cache_forwarded(self, tmp_path) -> None:
        from src.cache import JSONCache
        from src.generator import LiteLLMClient
        cache = JSONCache(str(tmp_path))
        llm = create_llm(cache=cache)
        assert isinstance(llm, LiteLLMClient)
        assert llm._cache is cache

    def test_no_cache_by_default(self) -> None:
        from src.generator import LiteLLMClient
        llm = create_llm()
        assert llm._cache is None
