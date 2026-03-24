"""Tests for FAISSVectorStore.

Uses real FAISS operations — no mocks needed for small data.
Tests: add+search, empty index, dimension mismatch, save/load roundtrip,
cosine score ≈ 1.0 for self-search, len(), isinstance BaseVectorStore.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.interfaces import BaseVectorStore
from src.schemas import Chunk, ChunkMetadata
from src.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Fixtures
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


def _unit_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Return n L2-normalised random float32 vectors of shape (n, dim)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ---------------------------------------------------------------------------
# Basic add + search
# ---------------------------------------------------------------------------


class TestAddAndSearch:
    def test_search_returns_correct_number(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
        vecs = _unit_vectors(5, 8)
        store.add(chunks, vecs)

        results = store.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_returns_chunk_and_float(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunk = _make_chunk()
        vec = _unit_vectors(1, 8)
        store.add([chunk], vec)

        results = store.search(vec[0], top_k=1)
        assert len(results) == 1
        c, score = results[0]
        assert isinstance(c, Chunk)
        assert isinstance(score, float)

    def test_self_search_score_near_one(self) -> None:
        store = FAISSVectorStore(dimension=16)
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        vecs = _unit_vectors(4, 16)
        store.add(chunks, vecs)

        results = store.search(vecs[2], top_k=1)
        assert len(results) == 1
        _, score = results[0]
        assert abs(score - 1.0) < 1e-5

    def test_results_sorted_descending(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        vecs = _unit_vectors(5, 8)
        store.add(chunks, vecs)

        results = store.search(vecs[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_index_returns_all(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        vecs = _unit_vectors(3, 8)
        store.add(chunks, vecs)

        results = store.search(vecs[0], top_k=100)
        assert len(results) == 3

    def test_search_preserves_chunk_content(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunk = _make_chunk("very specific content xyz")
        vec = _unit_vectors(1, 8)
        store.add([chunk], vec)

        results = store.search(vec[0], top_k=1)
        assert results[0][0].content == "very specific content xyz"


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


class TestEmptyIndex:
    def test_search_on_empty_returns_empty_list(self) -> None:
        store = FAISSVectorStore(dimension=8)
        query = _unit_vectors(1, 8)[0]
        assert store.search(query, top_k=5) == []

    def test_len_on_empty_is_zero(self) -> None:
        store = FAISSVectorStore(dimension=8)
        assert len(store) == 0


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_after_add(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"c{i}") for i in range(7)]
        vecs = _unit_vectors(7, 8)
        store.add(chunks, vecs)
        assert len(store) == 7

    def test_len_increments_on_multiple_adds(self) -> None:
        store = FAISSVectorStore(dimension=8)
        vecs = _unit_vectors(4, 8)
        store.add([_make_chunk(f"c{i}") for i in range(2)], vecs[:2])
        store.add([_make_chunk(f"c{i}") for i in range(2, 4)], vecs[2:])
        assert len(store) == 4


# ---------------------------------------------------------------------------
# Dimension mismatch
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    def test_wrong_embedding_dim_raises(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk()]
        wrong_vecs = _unit_vectors(1, 16)  # 16 != 8
        with pytest.raises(ValueError, match="dimension"):
            store.add(chunks, wrong_vecs)

    def test_chunks_embeddings_length_mismatch_raises(self) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(), _make_chunk()]  # 2 chunks
        vecs = _unit_vectors(3, 8)  # 3 embeddings
        with pytest.raises(ValueError):
            store.add(chunks, vecs)


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip_preserves_chunks(self, tmp_path) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"content {i}") for i in range(5)]
        vecs = _unit_vectors(5, 8)
        store.add(chunks, vecs)

        prefix = str(tmp_path / "index")
        store.save(prefix)

        store2 = FAISSVectorStore(dimension=8)
        store2.load(prefix)

        assert len(store2) == 5
        contents = {c.content for c in store2._chunks}
        for i in range(5):
            assert f"content {i}" in contents

    def test_save_creates_faiss_and_json_files(self, tmp_path) -> None:
        store = FAISSVectorStore(dimension=8)
        store.add([_make_chunk()], _unit_vectors(1, 8))

        prefix = str(tmp_path / "idx")
        store.save(prefix)

        assert (tmp_path / "idx.faiss").exists()
        assert (tmp_path / "idx.json").exists()

    def test_save_load_roundtrip_search_works(self, tmp_path) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        vecs = _unit_vectors(4, 8)
        store.add(chunks, vecs)

        prefix = str(tmp_path / "index")
        store.save(prefix)

        store2 = FAISSVectorStore(dimension=8)
        store2.load(prefix)

        results = store2.search(vecs[0], top_k=1)
        assert len(results) == 1
        _, score = results[0]
        assert abs(score - 1.0) < 1e-5

    def test_save_excludes_embedding_from_json(self, tmp_path) -> None:
        import json

        store = FAISSVectorStore(dimension=8)
        chunk = _make_chunk()
        chunk.embedding = _unit_vectors(1, 8)[0]  # set embedding on chunk
        store.add([chunk], _unit_vectors(1, 8))

        prefix = str(tmp_path / "index")
        store.save(prefix)

        data = json.loads((tmp_path / "index.json").read_text())
        assert "embedding" not in data[0]

    def test_load_restores_chunk_ids(self, tmp_path) -> None:
        store = FAISSVectorStore(dimension=8)
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        original_ids = [c.id for c in chunks]
        store.add(chunks, _unit_vectors(3, 8))

        prefix = str(tmp_path / "index")
        store.save(prefix)

        store2 = FAISSVectorStore(dimension=8)
        store2.load(prefix)

        loaded_ids = [c.id for c in store2._chunks]
        assert loaded_ids == original_ids


# ---------------------------------------------------------------------------
# isinstance check
# ---------------------------------------------------------------------------


class TestInterface:
    def test_implements_base_vector_store(self) -> None:
        store = FAISSVectorStore(dimension=8)
        assert isinstance(store, BaseVectorStore)
