"""Tests for MiniLMEmbedder and MpnetEmbedder.

SentenceTransformer is mocked to avoid loading 400MB+ models in CI.
Pattern matches test_chunkers.py where EmbeddingSemanticChunker mocks its model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.interfaces import BaseEmbedder


# ---------------------------------------------------------------------------
# Parametrize over both embedder classes so we don't duplicate test logic
# ---------------------------------------------------------------------------

EMBEDDER_PARAMS = [
    ("src.embedders.minilm.SentenceTransformer", "src.embedders.minilm.MiniLMEmbedder", 384),
    ("src.embedders.mpnet.SentenceTransformer", "src.embedders.mpnet.MpnetEmbedder", 768),
]


def _make_mock_model(dimension: int, n: int = 3) -> MagicMock:
    """Return a mock SentenceTransformer that produces unit-length random vectors."""
    mock = MagicMock()
    raw = np.random.randn(n, dimension).astype(np.float32)
    # L2-normalise so tests checking norms pass even before embedder normalises
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    mock.encode.return_value = raw / norms
    return mock


# ---------------------------------------------------------------------------
# Tests parametrized over both embedders
# ---------------------------------------------------------------------------


class TestEmbedderShape:
    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_returns_correct_shape(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim, n=3)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed(["a", "b", "c"])
            assert result.shape == (3, dim)

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_query_returns_1d(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim, n=1)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed_query("what is attention?")
            assert result.shape == (dim,)

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_l2_normalised(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            # Give the mock raw un-normalised vectors so we prove the embedder normalises them
            mock_model = MagicMock()
            raw = np.random.randn(4, dim).astype(np.float32) * 5.0  # scale up to ensure non-unit
            mock_model.encode.return_value = raw
            mock_st.return_value = mock_model

            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed(["a", "b", "c", "d"])
            norms = np.linalg.norm(result, axis=1)
            np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_query_l2_normalised(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_model = MagicMock()
            raw = np.random.randn(1, dim).astype(np.float32) * 3.0
            mock_model.encode.return_value = raw
            mock_st.return_value = mock_model

            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed_query("test query")
            assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_dimensions_property(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            assert embedder.dimensions == dim

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_implements_base_embedder(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            assert isinstance(embedder, BaseEmbedder)

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_empty_list_returns_empty_array(
        self, st_path: str, cls_path: str, dim: int
    ) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed([])
            assert result.shape == (0, dim)

    @pytest.mark.parametrize("st_path,cls_path,dim", EMBEDDER_PARAMS)
    def test_embed_returns_float32(self, st_path: str, cls_path: str, dim: int) -> None:
        import importlib

        with patch(st_path) as mock_st:
            mock_st.return_value = _make_mock_model(dim)
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            embedder = getattr(mod, cls_name)()

            result = embedder.embed(["test"])
            assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Specific dimension tests for documentation
# ---------------------------------------------------------------------------


class TestMiniLMDimension:
    def test_minilm_dimensions_is_384(self) -> None:
        with patch("src.embedders.minilm.SentenceTransformer") as mock_st:
            mock_st.return_value = _make_mock_model(384)
            from src.embedders.minilm import MiniLMEmbedder

            embedder = MiniLMEmbedder()
            assert embedder.dimensions == 384


class TestMpnetDimension:
    def test_mpnet_dimensions_is_768(self) -> None:
        with patch("src.embedders.mpnet.SentenceTransformer") as mock_st:
            mock_st.return_value = _make_mock_model(768)
            from src.embedders.mpnet import MpnetEmbedder

            embedder = MpnetEmbedder()
            assert embedder.dimensions == 768


# ---------------------------------------------------------------------------
# OpenAIEmbedder tests
# ---------------------------------------------------------------------------


def _make_litellm_response(n: int, dim: int = 1536) -> MagicMock:
    """Mock litellm.embedding() response with n unit vectors."""
    raw = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms

    mock_response = MagicMock()
    mock_response.data = [{"embedding": vectors[i].tolist()} for i in range(n)]
    return mock_response


class TestOpenAIEmbedder:
    def test_embed_returns_correct_shape(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        with patch("src.embedders.openai_embedder.litellm.embedding") as mock_embed:
            mock_embed.return_value = _make_litellm_response(3)
            embedder = OpenAIEmbedder()
            result = embedder.embed(["a", "b", "c"])

        assert result.shape == (3, 1536)

    def test_embed_empty_list_returns_empty_array(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        result = embedder.embed([])
        assert result.shape == (0, 1536)

    def test_embed_returns_float32(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        with patch("src.embedders.openai_embedder.litellm.embedding") as mock_embed:
            mock_embed.return_value = _make_litellm_response(2)
            embedder = OpenAIEmbedder()
            result = embedder.embed(["x", "y"])

        assert result.dtype == np.float32

    def test_embed_l2_normalised(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        # Return un-normalised vectors to prove the embedder normalises them
        raw = np.random.randn(4, 1536).astype(np.float32) * 5.0
        mock_response = MagicMock()
        mock_response.data = [{"embedding": raw[i].tolist()} for i in range(4)]

        with patch("src.embedders.openai_embedder.litellm.embedding", return_value=mock_response):
            embedder = OpenAIEmbedder()
            result = embedder.embed(["a", "b", "c", "d"])

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)

    def test_embed_query_returns_1d(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        with patch("src.embedders.openai_embedder.litellm.embedding") as mock_embed:
            mock_embed.return_value = _make_litellm_response(1)
            embedder = OpenAIEmbedder()
            result = embedder.embed_query("what is the return policy?")

        assert result.shape == (1536,)

    def test_embed_query_l2_normalised(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        raw = np.random.randn(1, 1536).astype(np.float32) * 3.0
        mock_response = MagicMock()
        mock_response.data = [{"embedding": raw[0].tolist()}]

        with patch("src.embedders.openai_embedder.litellm.embedding", return_value=mock_response):
            embedder = OpenAIEmbedder()
            result = embedder.embed_query("test query")

        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_dimensions_property(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        assert embedder.dimensions == 1536

    def test_implements_base_embedder(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        assert isinstance(embedder, BaseEmbedder)

    def test_150_texts_makes_2_api_calls(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        texts = [f"text {i}" for i in range(150)]

        with patch("src.embedders.openai_embedder.litellm.embedding") as mock_embed:
            mock_embed.side_effect = [
                _make_litellm_response(100),
                _make_litellm_response(50),
            ]
            embedder = OpenAIEmbedder()
            result = embedder.embed(texts)

        assert mock_embed.call_count == 2
        assert result.shape == (150, 1536)

    def test_batch_size_boundary_exact_100(self) -> None:
        from src.embedders.openai_embedder import OpenAIEmbedder

        texts = [f"t{i}" for i in range(100)]

        with patch("src.embedders.openai_embedder.litellm.embedding") as mock_embed:
            mock_embed.return_value = _make_litellm_response(100)
            embedder = OpenAIEmbedder()
            result = embedder.embed(texts)

        assert mock_embed.call_count == 1
        assert result.shape == (100, 1536)
