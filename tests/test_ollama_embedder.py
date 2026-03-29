"""Tests for OllamaEmbedder.

httpx is mocked throughout — no real Ollama server required.
Pattern mirrors test_embedders.py where SentenceTransformer is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedders.ollama_embedder import OllamaEmbedder, OllamaUnavailableError
from src.interfaces import BaseEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tags_response() -> MagicMock:
    """Simulate a successful GET /api/tags response."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"models": [{"name": "nomic-embed-text"}]}
    return resp


def _mock_embed_response(dim: int = 768) -> MagicMock:
    """Simulate a successful POST /api/embeddings response."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    vec = np.random.randn(dim).astype(float).tolist()
    resp.json.return_value = {"embedding": vec}
    return resp


def _make_embedder() -> OllamaEmbedder:
    """Create OllamaEmbedder with mocked health check."""
    with patch("httpx.get", return_value=_mock_tags_response()):
        return OllamaEmbedder()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_success_initialises_embedder(self) -> None:
        with patch("httpx.get", return_value=_mock_tags_response()):
            embedder = OllamaEmbedder()
        assert isinstance(embedder, OllamaEmbedder)

    def test_connect_error_raises_unavailable(self) -> None:
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OllamaUnavailableError, match="Ollama not available"):
                OllamaEmbedder()

    def test_timeout_raises_unavailable(self) -> None:
        import httpx
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(OllamaUnavailableError, match="Ollama not available"):
                OllamaEmbedder()

    def test_http_error_raises_unavailable(self) -> None:
        import httpx
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch("httpx.get", return_value=resp):
            with pytest.raises(OllamaUnavailableError):
                OllamaEmbedder()


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_embed_shape(self) -> None:
        embedder = _make_embedder()
        n = 4
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_embed_response(768)

        with patch("httpx.Client", return_value=mock_client):
            result = embedder.embed(["a", "b", "c", "d"])

        assert result.shape == (n, 768)
        assert result.dtype == np.float32

    def test_embed_l2_normalised(self) -> None:
        embedder = _make_embedder()
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_embed_response(768)

        with patch("httpx.Client", return_value=mock_client):
            result = embedder.embed(["hello", "world"])

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty_list_returns_empty_array(self) -> None:
        embedder = _make_embedder()
        result = embedder.embed([])
        assert result.shape == (0, 768)
        assert result.dtype == np.float32

    def test_embed_single_text(self) -> None:
        embedder = _make_embedder()
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_embed_response(768)

        with patch("httpx.Client", return_value=mock_client):
            result = embedder.embed(["single"])

        assert result.shape == (1, 768)


# ---------------------------------------------------------------------------
# embed_query()
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_embed_query_returns_1d(self) -> None:
        embedder = _make_embedder()
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_embed_response(768)

        with patch("httpx.Client", return_value=mock_client):
            result = embedder.embed_query("what is attention?")

        assert result.shape == (768,)
        assert result.dtype == np.float32

    def test_embed_query_is_normalised(self) -> None:
        embedder = _make_embedder()
        mock_client = MagicMock()
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_embed_response(768)

        with patch("httpx.Client", return_value=mock_client):
            result = embedder.embed_query("test query")

        assert abs(np.linalg.norm(result) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# dimensions property
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_dimensions_is_768(self) -> None:
        embedder = _make_embedder()
        assert embedder.dimensions == 768

    def test_is_base_embedder(self) -> None:
        embedder = _make_embedder()
        assert isinstance(embedder, BaseEmbedder)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_embedder_ollama_nomic(self) -> None:
        from src.factories import create_embedder
        with patch("httpx.get", return_value=_mock_tags_response()):
            embedder = create_embedder("ollama_nomic")
        assert isinstance(embedder, OllamaEmbedder)

    def test_create_embedder_ollama_nomic_is_base_embedder(self) -> None:
        from src.factories import create_embedder
        with patch("httpx.get", return_value=_mock_tags_response()):
            embedder = create_embedder("ollama_nomic")
        assert isinstance(embedder, BaseEmbedder)
