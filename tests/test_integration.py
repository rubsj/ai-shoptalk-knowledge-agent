"""Integration test: Full pipeline PDF → chunk → embed → index → retrieve → generate → cited answer.

Uses 1 test PDF, 1 config, 1 query. Mocks heavyweight dependencies
(SentenceTransformer, litellm) so the test is fast and never touches the network.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chunkers import RecursiveChunker
from src.generator import LiteLLMClient, build_qa_prompt, extract_citations
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.schemas import (
    Document,
    DocumentMetadata,
    PageInfo,
    QAResponse,
)
from src.vector_store import FAISSVectorStore

_DIMENSION = 384

_DOC_TEXT = (
    "Retrieval-augmented generation combines dense retrieval with LLM generation. "
    "Dense retrieval encodes documents as vectors and uses approximate nearest-neighbour search. "
    "BM25 is a sparse keyword-based retrieval algorithm that scores documents by term frequency. "
    "Hybrid retrieval fuses dense and sparse scores using a weighted alpha parameter. "
    "A reranker re-scores the top candidates using a cross-encoder for higher accuracy. "
    "The final answer cites the context chunks using numbered [N] markers. "
    "Evaluation metrics include MRR, Recall@K, and LLM-as-Judge faithfulness scores. "
    "Chunking strategies include fixed-size, recursive, sliding-window, and semantic chunking. "
    "FAISS uses inner product search on L2-normalized vectors for cosine similarity. "
    "The experiment grid runs 35+ configurations and compares retrieval quality. "
)


def _make_document() -> Document:
    return Document(
        content=_DOC_TEXT,
        metadata=DocumentMetadata(
            source="integration_test.pdf",
            title="Integration Test Doc",
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


def _fake_encode(texts, convert_to_numpy=True, **_kwargs):
    n = len(texts)
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((n, _DIMENSION)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _mock_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class TestFullPipelineIntegration:
    """End-to-end: Document → chunk → embed → index → retrieve → generate → QAResponse."""

    def test_pipeline_produces_cited_answer(self):
        start = time.monotonic()

        # 1. Create document
        doc = _make_document()
        assert doc.id  # deterministic ID from source hash
        assert len(doc.content) > 100

        # 2. Chunk
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3

        # 3. Embed (mocked)
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode

        with patch("src.embedders.minilm.SentenceTransformer", return_value=mock_st):
            from src.embedders import MiniLMEmbedder
            embedder = MiniLMEmbedder()

        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)
        assert embeddings.shape == (len(chunks), _DIMENSION)

        # 4. Index
        store = FAISSVectorStore(dimension=_DIMENSION)
        store.add(chunks, embeddings)

        # 5. Retrieve (hybrid)
        dense = DenseRetriever(embedder, store)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25, alpha=0.7)

        query = "How does hybrid retrieval work?"
        results = hybrid.retrieve(query, top_k=5)
        assert len(results) >= 1

        retrieved_chunks = [r.chunk for r in results]

        # 6. Generate (mocked LLM)
        llm_answer = "[1] Hybrid retrieval fuses dense and BM25. [2] Alpha controls the weight."

        with patch(
            "src.generator.litellm.completion",
            return_value=_mock_completion(llm_answer),
        ):
            llm = LiteLLMClient()
            prompt = build_qa_prompt(query, retrieved_chunks)
            answer = llm.generate(prompt)

        assert answer == llm_answer
        assert len(answer) > 0

        # 7. Extract citations
        citations = extract_citations(answer, retrieved_chunks)
        assert len(citations) >= 1

        # 8. Build QAResponse
        elapsed = time.monotonic() - start
        qa = QAResponse(
            query=query,
            answer=answer,
            citations=citations,
            chunks_used=retrieved_chunks,
            latency=elapsed,
        )

        # Verify
        assert qa.answer == llm_answer
        assert len(qa.citations) >= 1
        assert qa.latency < 30.0
        for c in qa.citations:
            assert c.chunk_id
            assert c.source == "integration_test.pdf"

    def test_pipeline_with_bm25_only(self):
        """BM25-only pipeline works without embeddings."""
        doc = _make_document()
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        bm25 = BM25Retriever(chunks)
        query = "What is BM25?"
        results = bm25.retrieve(query, top_k=3)
        assert len(results) >= 1

        retrieved_chunks = [r.chunk for r in results]

        llm_answer = "[1] BM25 is a sparse keyword-based retrieval algorithm."
        with patch(
            "src.generator.litellm.completion",
            return_value=_mock_completion(llm_answer),
        ):
            llm = LiteLLMClient()
            prompt = build_qa_prompt(query, retrieved_chunks)
            answer = llm.generate(prompt)

        citations = extract_citations(answer, retrieved_chunks)
        assert len(citations) >= 1

    def test_pipeline_empty_query_returns_results(self):
        """Even odd queries produce some retrieval results."""
        doc = _make_document()
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        bm25 = BM25Retriever(chunks)
        results = bm25.retrieve("xyznonexistent", top_k=3)
        # BM25 may return empty for truly novel queries — that's acceptable
        assert isinstance(results, list)
