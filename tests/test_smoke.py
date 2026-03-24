"""End-to-end smoke test: Document → chunks → embed → index → retrieve → rerank → generate → citations.

Mocks out heavyweight dependencies (SentenceTransformer, CrossEncoder, litellm)
so the test is fast and never touches the network.

Pipeline under test:
  Document
    → FixedSizeChunker             (real)
    → MiniLMEmbedder               (SentenceTransformer mocked)
    → FAISSVectorStore             (real — fast on tiny data)
    → HybridRetriever              (real — uses real Dense + BM25 sub-retrievers)
    → CrossEncoderReranker         (CrossEncoder mocked)
    → LiteLLMClient                (litellm.completion mocked, cache enabled)
    → extract_citations            (real)
    → verify citations map back to chunks
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cache import JSONCache
from src.chunkers import FixedSizeChunker
from src.generator import LiteLLMClient, build_qa_prompt, extract_citations
from src.rerankers import CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.schemas import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    PageInfo,
)
from src.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIMENSION = 384
_DOC_TEXT = (
    "Retrieval-augmented generation combines dense retrieval with LLM generation. "
    "Dense retrieval encodes documents as vectors and uses approximate nearest-neighbour search. "
    "BM25 is a sparse keyword-based retrieval algorithm that scores documents by term frequency. "
    "Hybrid retrieval fuses dense and sparse scores using a weighted alpha parameter. "
    "A reranker re-scores the top candidates using a cross-encoder for higher accuracy. "
    "The final answer cites the context chunks using numbered [N] markers. "
    "Evaluation metrics include MRR, Recall@K, and LLM-as-Judge faithfulness scores. "
    "Chunking strategies include fixed-size, recursive, sliding-window, and semantic chunking."
)


def _make_document() -> Document:
    return Document(
        content=_DOC_TEXT,
        metadata=DocumentMetadata(
            source="smoke_test.pdf",
            title="Smoke Test Document",
            author="Test",
            page_count=1,
        ),
        pages=[
            PageInfo(
                page_number=0,
                text=_DOC_TEXT,
                char_count=len(_DOC_TEXT),
            )
        ],
    )


def _fake_encode(texts, convert_to_numpy=True):
    """Return deterministic unit vectors — one per text."""
    n = len(texts)
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, _DIMENSION)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _mock_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmokeEndToEnd:
    """Full pipeline with mocked heavyweight dependencies."""

    @pytest.fixture()
    def pipeline(self, tmp_path):
        """Build and return all pipeline components (embedder mocked)."""
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode

        with patch("src.embedders.minilm.SentenceTransformer", return_value=mock_st):
            from src.embedders import MiniLMEmbedder
            embedder = MiniLMEmbedder()

        # Chunk the document
        doc = _make_document()
        chunker = FixedSizeChunker(chunk_size=120, chunk_overlap=10)
        chunks: list[Chunk] = chunker.chunk(doc)
        assert len(chunks) >= 2, "Need ≥2 chunks for citation test"

        # Embed and index
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)
        store = FAISSVectorStore(dimension=_DIMENSION)
        store.add(chunks, embeddings)

        # Retrievers
        dense = DenseRetriever(embedder, store)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, alpha=0.7)

        return {"chunks": chunks, "hybrid": hybrid, "embedder": embedder}

    def test_pipeline_returns_two_citations(self, pipeline, tmp_path):
        """LLM answer citing [1] and [2] → exactly 2 Citation objects."""
        query = "How does hybrid retrieval work?"

        # Retrieve + rerank (CrossEncoder mocked)
        hybrid = pipeline["hybrid"]
        retrieval_results = hybrid.retrieve(query, top_k=4)
        assert len(retrieval_results) >= 2

        context_chunks = [r.chunk for r in retrieval_results]

        with patch(
            "src.rerankers.cross_encoder.CrossEncoder"
        ) as mock_ce_cls:
            mock_ce = MagicMock()
            mock_ce_cls.return_value = mock_ce
            mock_ce.predict.return_value = np.array(
                [float(i) for i in range(len(context_chunks), 0, -1)]
            )
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(query, retrieval_results, top_k=2)

        assert len(reranked) == 2
        final_chunks = [r.chunk for r in reranked]

        # Generate with mocked LiteLLM
        llm_answer = "[1] Hybrid retrieval fuses dense and BM25. [2] Alpha controls the weight."
        cache = JSONCache(str(tmp_path))

        with patch(
            "src.generator.litellm.completion",
            return_value=_mock_completion(llm_answer),
        ):
            llm = LiteLLMClient(cache=cache)
            prompt = build_qa_prompt(query, final_chunks)
            answer = llm.generate(prompt)

        assert answer == llm_answer

        citations = extract_citations(answer, final_chunks)
        assert len(citations) == 2
        assert citations[0].chunk_id == final_chunks[0].id
        assert citations[1].chunk_id == final_chunks[1].id
        for c in citations:
            assert c.source == "smoke_test.pdf"
            assert c.relevance_score == 0.0

    def test_cache_deduplicates_llm_calls(self, pipeline, tmp_path):
        """Same query twice → litellm.completion called exactly once."""
        query = "What is BM25?"
        hybrid = pipeline["hybrid"]
        retrieval_results = hybrid.retrieve(query, top_k=2)
        context_chunks = [r.chunk for r in retrieval_results]
        prompt = build_qa_prompt(query, context_chunks)

        cache = JSONCache(str(tmp_path))
        llm_answer = "[1] BM25 is a sparse retrieval algorithm."

        with patch(
            "src.generator.litellm.completion",
            return_value=_mock_completion(llm_answer),
        ) as mock_comp:
            llm = LiteLLMClient(cache=cache)
            answer_1 = llm.generate(prompt, system_prompt="You are helpful.")
            answer_2 = llm.generate(prompt, system_prompt="You are helpful.")

        mock_comp.assert_called_once()
        assert answer_1 == answer_2 == llm_answer

    def test_chunk_ids_are_valid_in_citations(self, pipeline, tmp_path):
        """Every citation chunk_id must correspond to an actual chunk."""
        query = "What is dense retrieval?"
        hybrid = pipeline["hybrid"]
        results = hybrid.retrieve(query, top_k=3)
        context_chunks = [r.chunk for r in results]
        prompt = build_qa_prompt(query, context_chunks)

        known_ids = {c.id for c in context_chunks}
        llm_answer = "[1] Dense retrieval uses vector similarity. [2] It requires an embedder. [3] FAISS speeds up search."

        with patch(
            "src.generator.litellm.completion",
            return_value=_mock_completion(llm_answer),
        ):
            llm = LiteLLMClient()
            answer = llm.generate(prompt)

        citations = extract_citations(answer, context_chunks)
        for c in citations:
            assert c.chunk_id in known_ids
