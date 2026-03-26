"""Pydantic data models for the ShopTalk Knowledge Agent.

No raw dicts cross module boundaries — everything typed here.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases — Literal so Pydantic validates at runtime (like Java enums, lighter)
# ---------------------------------------------------------------------------

RetrieverType = Literal["dense", "bm25", "hybrid"]

ChunkingStrategy = Literal[
    "fixed",
    "recursive",
    "sliding_window",
    "heading_semantic",
    "embedding_semantic",
]

EmbeddingModel = Literal["minilm", "mpnet", "openai"]

RerankerType = Literal["cohere", "cross_encoder"]


# ---------------------------------------------------------------------------
# Core document models
# ---------------------------------------------------------------------------


class PageInfo(BaseModel):
    """Single PDF page text + metadata."""

    page_number: int = Field(..., ge=0, description="0-indexed page number")
    text: str = Field(..., description="Cleaned text content of the page")
    char_count: int = Field(..., ge=0, description="Character count after cleaning")


class DocumentMetadata(BaseModel):
    """Bibliographic metadata from PDF."""

    source: str = Field(..., min_length=1, description="File path or URL of the PDF")
    title: str = Field(default="", description="Document title (from PDF metadata)")
    author: str = Field(default="", description="Document author(s)")
    page_count: int = Field(..., ge=1, description="Total number of pages")


class Document(BaseModel):
    """Fully extracted PDF, ready for chunking."""

    # ID derived from source path so re-extraction produces the same ID.
    # WHY: ground truth chunk IDs encode document_id; deterministic doc IDs
    # mean chunk IDs are stable across runs with the same PDF.
    id: str = Field(default="", description="Unique document ID (derived from metadata.source)")
    content: str = Field(..., min_length=1, description="Full text content (all pages joined with \\n\\n)")
    metadata: DocumentMetadata = Field(..., description="Bibliographic metadata")
    pages: list[PageInfo] = Field(..., min_length=1, description="Per-page text and metadata")

    @model_validator(mode="after")
    def derive_id_from_source(self) -> Document:
        """Set id = md5(source)[:16] if not explicitly provided."""
        if not self.id:
            self.id = hashlib.md5(self.metadata.source.encode()).hexdigest()[:16]
        return self


# ---------------------------------------------------------------------------
# Chunk models
# ---------------------------------------------------------------------------


class ChunkMetadata(BaseModel):
    """Provenance — links a chunk back to its source document."""

    document_id: str = Field(..., min_length=1, description="ID of the parent Document")
    source: str = Field(..., min_length=1, description="Source file path (copied from DocumentMetadata)")
    page_number: int = Field(..., ge=0, description="Page where this chunk starts")
    start_char: int = Field(..., ge=0, description="Start character offset in Document.content")
    end_char: int = Field(..., ge=0, description="End character offset in Document.content (exclusive)")
    chunk_index: int = Field(..., ge=0, description="Sequential index of this chunk within the document")

    @model_validator(mode="after")
    def end_must_be_gte_start(self) -> ChunkMetadata:
        """Zero-length (start == end) is degenerate but allowed; only reject end < start."""
        if self.end_char < self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be >= start_char ({self.start_char})"
            )
        return self


class Chunk(BaseModel):
    """Atomic retrieval unit — text slice with provenance and optional embedding."""

    # arbitrary_types_allowed: lets us store np.ndarray without serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Provenance — links back to source document")
    embedding: np.ndarray | None = Field(
        default=None,
        description="Dense embedding vector (set after encoding, None before)",
    )


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """Retrieved chunk + score + retriever metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: Chunk = Field(..., description="The retrieved chunk")
    score: float = Field(..., description="Similarity or relevance score (higher = more relevant)")
    retriever_type: RetrieverType = Field(..., description="Which retriever produced this result")
    rank: int = Field(..., ge=1, description="Rank in the result list (1 = highest ranked)")


class Citation(BaseModel):
    """Source reference parsed from [N] markers in generated answers."""

    chunk_id: str = Field(..., min_length=1, description="ID of the Chunk this citation refers to")
    source: str = Field(..., min_length=1, description="Source file path")
    page_number: int = Field(..., ge=0, description="Page number in the source document")
    text_snippet: str = Field(..., min_length=1, description="Quoted text from the chunk")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to the query [0, 1]")


class QAResponse(BaseModel):
    """Full query output: answer, citations, perf metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(..., min_length=1, description="The original user query")
    answer: str = Field(..., min_length=1, description="Generated answer with [N] citation markers")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Parsed citations referenced in the answer",
    )
    chunks_used: list[Chunk] = Field(..., description="Top-K chunks passed to the LLM")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score from the LLM",
    )
    latency: float = Field(..., ge=0.0, description="End-to-end query latency in seconds")


# ---------------------------------------------------------------------------
# Experiment models
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """One experiment run spec — all config dimensions + cross-field validators.

    Pydantic validates at load time so we catch bad combos (e.g., hybrid
    without alpha) before burning API credits on the experiment grid.
    """

    chunking_strategy: ChunkingStrategy = Field(..., description="Which chunking strategy to use")
    chunk_size: int = Field(default=512, ge=50, le=5000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between consecutive chunks in characters")
    embedding_model: EmbeddingModel | None = Field(
        default=None,
        description="Embedding model (None for bm25-only configs)",
    )
    retriever_type: RetrieverType = Field(..., description="Retrieval method")
    hybrid_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Dense weight in hybrid retrieval [0,1]. Required iff retriever_type=='hybrid'.",
    )
    use_reranking: bool = Field(default=False, description="Whether to apply a reranker on top-K results")
    reranker_type: RerankerType | None = Field(
        default=None,
        description="Reranker to use. Required iff use_reranking==True.",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of chunks to retrieve")
    window_size_tokens: int | None = Field(
        default=None,
        ge=10,
        description="Token window size (required for sliding_window strategy)",
    )
    step_size_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Token step size (required for sliding_window strategy)",
    )
    breakpoint_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold below which a chunk boundary is inserted",
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="Minimum chunk size in characters (small chunks are merged with neighbour)",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info: object) -> int:
        """overlap >= size means every chunk is 100% duplicate of the previous one."""
        chunk_size = getattr(info, "data", {}).get("chunk_size", 512)  # type: ignore[arg-type]
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @model_validator(mode="after")
    def hybrid_requires_alpha(self) -> ExperimentConfig:
        """Min-max score fusion needs an explicit alpha — no silent default."""
        if self.retriever_type == "hybrid" and self.hybrid_alpha is None:
            raise ValueError("hybrid_alpha is required when retriever_type='hybrid'")
        if self.retriever_type != "hybrid" and self.hybrid_alpha is not None:
            raise ValueError("hybrid_alpha must be None when retriever_type != 'hybrid'")
        return self

    @model_validator(mode="after")
    def reranking_requires_type(self) -> ExperimentConfig:
        """use_reranking=True with no reranker_type would silently skip reranking."""
        if self.use_reranking and self.reranker_type is None:
            raise ValueError("reranker_type is required when use_reranking=True")
        if not self.use_reranking and self.reranker_type is not None:
            raise ValueError("reranker_type must be None when use_reranking=False")
        return self

    @model_validator(mode="after")
    def dense_hybrid_requires_embedding_model(self) -> ExperimentConfig:
        """BM25 is lexical only. Dense/hybrid need an embedding model."""
        if self.retriever_type in ("dense", "hybrid") and self.embedding_model is None:
            raise ValueError(
                f"embedding_model is required when retriever_type='{self.retriever_type}'"
            )
        if self.retriever_type == "bm25" and self.embedding_model is not None:
            raise ValueError("embedding_model must be None when retriever_type='bm25'")
        return self

    @model_validator(mode="after")
    def sliding_window_requires_params(self) -> ExperimentConfig:
        """Can't run sliding window without window_size and step_size."""
        if self.chunking_strategy == "sliding_window":
            if self.window_size_tokens is None or self.step_size_tokens is None:
                raise ValueError(
                    "window_size_tokens and step_size_tokens are required for sliding_window strategy"
                )
        return self


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics computed against ground truth."""

    recall_at_5: float = Field(..., ge=0.0, le=1.0, description="Recall@5 over the ground truth set")
    precision_at_5: float = Field(..., ge=0.0, le=1.0, description="Precision@5 over the ground truth set")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg_at_5: float = Field(..., ge=0.0, le=1.0, description="NDCG@5 with 4-level grading scheme")


class JudgeScores(BaseModel):
    """5-axis LLM-as-Judge scores for a batch of answers."""

    avg_relevance: float = Field(..., ge=1.0, le=5.0, description="Average relevance score (1-5)")
    avg_accuracy: float = Field(..., ge=1.0, le=5.0, description="Average factual accuracy score (1-5)")
    avg_completeness: float = Field(..., ge=1.0, le=5.0, description="Average answer completeness score (1-5)")
    avg_conciseness: float = Field(..., ge=1.0, le=5.0, description="Average conciseness score (1-5)")
    avg_citation_quality: float = Field(..., ge=1.0, le=5.0, description="Average citation quality score (1-5)")
    overall_average: float = Field(..., ge=1.0, le=5.0, description="Mean of the 5 axis scores")


class JudgeResult(BaseModel):
    """Single answer score from 5-axis LLM-as-Judge — Instructor response model."""

    relevance: int = Field(..., ge=1, le=5, description="Does the answer address the question? (1-5)")
    accuracy: int = Field(..., ge=1, le=5, description="Are all claims verifiable in the context? (1-5)")
    completeness: int = Field(..., ge=1, le=5, description="Is the answer thorough? (1-5)")
    conciseness: int = Field(..., ge=1, le=5, description="Is the answer appropriately brief? (1-5)")
    citation_quality: int = Field(..., ge=1, le=5, description="Are [N] sources properly attributed? (1-5)")


class PerformanceMetrics(BaseModel):
    """System perf metrics from an experiment run."""

    ingestion_time_seconds: float = Field(..., ge=0.0, description="Wall-clock time for PDF→index pipeline")
    avg_query_latency_ms: float = Field(..., ge=0.0, description="Average end-to-end query latency in milliseconds")
    index_size_bytes: int = Field(..., ge=0, description="FAISS index size on disk in bytes")
    peak_memory_mb: float = Field(..., ge=0.0, description="Peak RSS memory during the run in megabytes")
    embedding_source: str = Field(..., description="'local', 'api', or 'none' (BM25-only configs)")
    cost_estimate_usd: float = Field(default=0.0, ge=0.0, description="Estimated API cost in USD (0.0 for local/BM25)")


class QueryResult(BaseModel):
    """Per-query breakdown — enables difficulty analysis and per-axis aggregation."""

    query_id: str = Field(..., min_length=1, description="Matches GroundTruthQuery.query_id")
    question: str = Field(..., min_length=1, description="The evaluation question")
    answer: str = Field(..., min_length=1, description="LLM-generated answer for this query")
    retrieved_chunk_ids: list[str] = Field(default_factory=list, description="IDs of top-K retrieved chunks")
    retrieval_scores: RetrievalMetrics = Field(..., description="Per-query retrieval metrics (not averaged)")
    judge_result: JudgeResult | None = Field(default=None, description="Per-query judge scores (None if judge skipped)")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end latency for this query in milliseconds")


class ExperimentResult(BaseModel):
    """Full result: config + retrieval metrics + judge scores + perf."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_id: str = Field(..., min_length=1, description="Unique identifier for this experiment run")
    config: ExperimentConfig = Field(..., description="The experiment configuration that produced these results")
    metrics: RetrievalMetrics = Field(..., description="Retrieval quality metrics vs. ground truth")
    judge_scores: JudgeScores | None = Field(
        default=None,
        description="LLM-as-Judge scores (None if judge not run for this config)",
    )
    performance: PerformanceMetrics = Field(..., description="System performance metrics")
    query_results: list[QueryResult] = Field(default_factory=list, description="Per-query QA results for difficulty analysis")


# ---------------------------------------------------------------------------
# Ground truth models
# ---------------------------------------------------------------------------


class GroundTruthChunk(BaseModel):
    """Single chunk relevance judgment within a ground truth query."""

    chunk_id: str = Field(..., min_length=1, description="ID of the relevant Chunk")
    relevance_grade: int = Field(
        ..., ge=0, le=3,
        description="0=irrelevant, 1=same doc, 2=same section, 3=gold (directly answers)",
    )


class GroundTruthQuery(BaseModel):
    """One ground truth query with its relevant chunks and grades."""

    query_id: str = Field(..., min_length=1, description="Unique query identifier")
    question: str = Field(..., min_length=1, description="The evaluation question")
    relevant_chunks: list[GroundTruthChunk] = Field(
        ..., min_length=1, description="Chunks with relevance grades (grade >= 1)"
    )


class GroundTruthSet(BaseModel):
    """Complete ground truth dataset for evaluation."""

    queries: list[GroundTruthQuery] = Field(..., min_length=1, description="Curated evaluation queries")


class GeneratedQAPair(BaseModel):
    """Instructor response model for ground truth generation — one QA pair per chunk batch."""

    question: str = Field(..., min_length=1, description="Question answerable by 1-3 chunks in the batch")
    relevant_chunks: list[GroundTruthChunk] = Field(
        ..., min_length=1, description="Chunks with grades (only grade >= 1 included)"
    )