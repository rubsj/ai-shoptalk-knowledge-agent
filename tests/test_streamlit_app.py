"""Tests for src/streamlit_app.py helper functions.

Tests only the pure helper functions (build_config_from_ui, run_query) — these
are extracted specifically to be testable without a running Streamlit server.
All external dependencies (embedder, retriever, LLM) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.schemas import Citation, Chunk, ChunkMetadata, ExperimentConfig
from src.streamlit_app import build_config_from_ui, run_query


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "test content", source: str = "doc.pdf", page: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="doc1",
            source=source,
            page_number=page,
            start_char=0,
            end_char=len(content),
            chunk_index=0,
        ),
    )


# ---------------------------------------------------------------------------
# build_config_from_ui
# ---------------------------------------------------------------------------


class TestBuildConfigFromUi:
    def test_valid_dense_config(self):
        cfg = build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=5,
            use_reranking=False,
            reranker_type=None,
        )
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.chunking_strategy == "fixed"
        assert cfg.embedding_model == "minilm"
        assert cfg.retriever_type == "dense"
        assert cfg.top_k == 5

    def test_bm25_sets_embedding_model_none(self):
        cfg = build_config_from_ui(
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model=None,
            retriever_type="bm25",
            hybrid_alpha=None,
            top_k=3,
            use_reranking=False,
            reranker_type=None,
        )
        assert cfg.embedding_model is None
        assert cfg.retriever_type == "bm25"

    def test_hybrid_requires_alpha(self):
        """hybrid without alpha raises ValidationError from ExperimentConfig."""
        with pytest.raises(Exception):
            build_config_from_ui(
                chunking_strategy="fixed",
                chunk_size=512,
                chunk_overlap=50,
                embedding_model="minilm",
                retriever_type="hybrid",
                hybrid_alpha=None,  # missing — should fail
                top_k=5,
                use_reranking=False,
                reranker_type=None,
            )

    def test_hybrid_with_alpha_succeeds(self):
        cfg = build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="hybrid",
            hybrid_alpha=0.7,
            top_k=5,
            use_reranking=False,
            reranker_type=None,
        )
        assert cfg.hybrid_alpha == 0.7

    def test_reranking_requires_type(self):
        """use_reranking=True without reranker_type raises ValidationError."""
        with pytest.raises(Exception):
            build_config_from_ui(
                chunking_strategy="fixed",
                chunk_size=512,
                chunk_overlap=50,
                embedding_model="minilm",
                retriever_type="dense",
                hybrid_alpha=None,
                top_k=5,
                use_reranking=True,
                reranker_type=None,  # missing — should fail
            )

    def test_reranking_with_type_succeeds(self):
        cfg = build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=5,
            use_reranking=True,
            reranker_type="cross_encoder",
        )
        assert cfg.use_reranking is True
        assert cfg.reranker_type == "cross_encoder"

    def test_sliding_window_with_token_params(self):
        cfg = build_config_from_ui(
            chunking_strategy="sliding_window",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=5,
            use_reranking=False,
            reranker_type=None,
            window_size_tokens=200,
            step_size_tokens=100,
        )
        assert cfg.window_size_tokens == 200
        assert cfg.step_size_tokens == 100

    def test_top_k_propagated(self):
        cfg = build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=10,
            use_reranking=False,
            reranker_type=None,
        )
        assert cfg.top_k == 10


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------


class TestRunQuery:
    def _make_cfg(self) -> ExperimentConfig:
        return build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=3,
            use_reranking=False,
            reranker_type=None,
        )

    def test_returns_answer_and_citations(self):
        cfg = self._make_cfg()
        chunk = _make_chunk("ShopTalk ships in 2 days.")
        cit = Citation(
            chunk_id=chunk.id,
            source="doc.pdf",
            page_number=0,
            text_snippet="ShopTalk ships",
            relevance_score=0.0,
        )

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [MagicMock(chunk=chunk)]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Ships in 2 days [1]."

        mock_embedder = MagicMock()
        mock_store = MagicMock()

        with patch("src.streamlit_app.create_retriever", return_value=mock_retriever):
            with patch("src.streamlit_app.create_llm", return_value=mock_llm):
                with patch("src.streamlit_app.build_qa_prompt", return_value="prompt"):
                    with patch("src.streamlit_app.extract_citations", return_value=[cit]):
                        result = run_query(
                            query="How fast does ShopTalk ship?",
                            config=cfg,
                            embedder=mock_embedder,
                            chunks=[chunk],
                            vector_store=mock_store,
                        )

        assert result["answer"] == "Ships in 2 days [1]."
        assert len(result["citations"]) == 1
        assert result["citations"][0].chunk_id == chunk.id
        assert result["latency_ms"] >= 0
        assert chunk in result["chunks_used"]

    def test_no_citations_returns_empty_list(self):
        cfg = self._make_cfg()
        chunk = _make_chunk()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [MagicMock(chunk=chunk)]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "No citations here."

        with patch("src.streamlit_app.create_retriever", return_value=mock_retriever):
            with patch("src.streamlit_app.create_llm", return_value=mock_llm):
                with patch("src.streamlit_app.build_qa_prompt", return_value="prompt"):
                    with patch("src.streamlit_app.extract_citations", return_value=[]):
                        result = run_query(
                            query="anything",
                            config=cfg,
                            embedder=MagicMock(),
                            chunks=[chunk],
                            vector_store=MagicMock(),
                        )

        assert result["citations"] == []
        assert result["answer"] == "No citations here."

    def test_reranker_called_when_use_reranking_true(self):
        cfg = build_config_from_ui(
            chunking_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="minilm",
            retriever_type="dense",
            hybrid_alpha=None,
            top_k=3,
            use_reranking=True,
            reranker_type="cross_encoder",
        )
        chunk = _make_chunk()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [MagicMock(chunk=chunk)]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [MagicMock(chunk=chunk)]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Reranked answer."

        with patch("src.streamlit_app.create_retriever", return_value=mock_retriever):
            with patch("src.streamlit_app.create_reranker", return_value=mock_reranker):
                with patch("src.streamlit_app.create_llm", return_value=mock_llm):
                    with patch("src.streamlit_app.build_qa_prompt", return_value="prompt"):
                        with patch("src.streamlit_app.extract_citations", return_value=[]):
                            result = run_query(
                                query="test",
                                config=cfg,
                                embedder=MagicMock(),
                                chunks=[chunk],
                                vector_store=MagicMock(),
                            )

        mock_reranker.rerank.assert_called_once()
        assert result["answer"] == "Reranked answer."

    def test_latency_ms_is_positive(self):
        cfg = self._make_cfg()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "answer"

        with patch("src.streamlit_app.create_retriever", return_value=mock_retriever):
            with patch("src.streamlit_app.create_llm", return_value=mock_llm):
                with patch("src.streamlit_app.build_qa_prompt", return_value="prompt"):
                    with patch("src.streamlit_app.extract_citations", return_value=[]):
                        result = run_query(
                            query="timing?",
                            config=cfg,
                            embedder=MagicMock(),
                            chunks=[],
                            vector_store=MagicMock(),
                        )

        assert result["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------


class TestStreamlitAppImport:
    def test_module_imports_without_error(self):
        """Importing streamlit_app must not raise even if Streamlit is absent."""
        import src.streamlit_app  # noqa: F401

    def test_build_config_from_ui_is_callable(self):
        from src.streamlit_app import build_config_from_ui
        assert callable(build_config_from_ui)

    def test_run_query_is_callable(self):
        from src.streamlit_app import run_query
        assert callable(run_query)
