"""Ollama nomic-embed-text embedder — 768d local embeddings via REST API.

Zero cost, no network dependency (runs locally via Ollama).
API: POST http://localhost:11434/api/embeddings {"model": "nomic-embed-text", "prompt": text}
Health check: GET http://localhost:11434/api/tags (fail fast if Ollama is down).

WHY httpx over requests: async-ready, better timeout granularity, same API surface.
WHY sequential (not batched): Ollama /api/embeddings accepts one prompt per call.
WHY faiss.normalize_L2: consistent with MiniLMEmbedder/MpnetEmbedder — all embedders
return unit vectors so FAISSVectorStore (IndexFlatIP) computes cosine similarity.
"""

from __future__ import annotations

import logging
import os

import httpx
import numpy as np

from src.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama server is not reachable at startup. Callers catch and skip."""


class OllamaEmbedder(BaseEmbedder):
    """768-dimensional embedder using nomic-embed-text via Ollama REST API.

    Runs fully locally — zero API cost, no data leaves the machine.
    Same 768d as MpnetEmbedder so FAISS index infrastructure is reused.

    WHY health check in __init__: fail fast at config load time rather than
    mid-experiment. Experiment runner catches OllamaUnavailableError and skips
    the ollama_nomic group gracefully.
    """

    _MODEL_NAME = "nomic-embed-text"
    _DIMENSIONS = 768
    _DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, device: str | None = None) -> None:
        # device param accepted for factory API compatibility — Ollama manages
        # its own device (CPU/GPU), Python side has no control.
        self._base_url = os.getenv("OLLAMA_BASE_URL", self._DEFAULT_BASE_URL)

        # Health check — fail fast if Ollama is not running
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            logger.info(
                "OllamaEmbedder ready — %s, base_url=%s", self._MODEL_NAME, self._base_url
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise OllamaUnavailableError(
                f"Ollama not available at {self._base_url}: {e}"
            ) from e

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed texts via sequential Ollama REST calls.

        Returns shape (len(texts), 768), L2-normalised.
        Sequential because /api/embeddings accepts one prompt per call.
        """
        if not texts:
            return np.empty((0, self._DIMENSIONS), dtype=np.float32)

        vectors: list[list[float]] = []
        with httpx.Client(timeout=30.0) as client:
            for i, text in enumerate(texts):
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._MODEL_NAME, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
                if (i + 1) % 50 == 0:
                    logger.info("OllamaEmbedder: embedded %d/%d texts", i + 1, len(texts))

        embeddings = np.array(vectors, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings /= norms
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Single query embed. Returns shape (768,), L2-normalised."""
        # WHY: reuse embed() so normalization path is identical for docs and queries
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        """768 — FAISS IndexFlatIP must be initialized with this value."""
        return self._DIMENSIONS
