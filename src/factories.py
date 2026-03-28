"""Factory: YAML config strings → class instances.

Maps config values to concrete implementations:
  {"chunker": "recursive"} → RecursiveChunker(config)
  {"embedder": "minilm"}   → MiniLMEmbedder(config)
  {"retriever": "hybrid"}  → HybridRetriever(dense, bm25, alpha)

Like Spring's @Component + @Qualifier. Callers never import concrete classes.
"""

from __future__ import annotations

from src.cache import JSONCache
from src.chunkers import (
    EmbeddingSemanticChunker,
    FixedSizeChunker,
    HeadingSemanticChunker,
    RecursiveChunker,
    SlidingWindowChunker,
)
from src.embedders import MiniLMEmbedder, MpnetEmbedder
from src.interfaces import BaseChunker, BaseEmbedder, BaseLLM, BaseReranker, BaseRetriever
from src.rerankers import CohereReranker, CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.schemas import Chunk, ExperimentConfig


def create_chunker(config: ExperimentConfig) -> BaseChunker:
    """Return the chunker specified by config.chunking_strategy."""
    strategy = config.chunking_strategy
    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if strategy == "sliding_window":
        return SlidingWindowChunker(
            window_size=config.window_size_tokens,
            step_size=config.step_size_tokens,
        )
    if strategy == "heading_semantic":
        return HeadingSemanticChunker(
            min_chunk_size=config.min_chunk_size,
        )
    if strategy == "embedding_semantic":
        return EmbeddingSemanticChunker(
            breakpoint_threshold=config.breakpoint_threshold,
            min_chunk_size=config.min_chunk_size,
        )
    raise ValueError(f"Unknown chunking_strategy: {strategy!r}")


def create_embedder(model_name: str, device: str | None = None) -> BaseEmbedder:
    """Return the embedder for the given model name ('minilm', 'mpnet', or 'openai')."""
    if model_name == "minilm":
        return MiniLMEmbedder(device=device)
    if model_name == "mpnet":
        return MpnetEmbedder(device=device)
    if model_name == "openai":
        from src.embedders.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedding model: {model_name!r}")


def create_retriever(
    config: ExperimentConfig,
    embedder: BaseEmbedder | None,
    chunks: list[Chunk],
    vector_store,
) -> BaseRetriever:
    """Return the retriever specified by config.retriever_type.

    For hybrid: builds both dense and BM25 sub-retrievers internally.
    """
    retriever_type = config.retriever_type
    if retriever_type == "bm25":
        return BM25Retriever(chunks)
    if retriever_type == "dense":
        return DenseRetriever(embedder, vector_store)
    if retriever_type == "hybrid":
        dense = DenseRetriever(embedder, vector_store)
        bm25 = BM25Retriever(chunks)
        alpha = config.hybrid_alpha if config.hybrid_alpha is not None else 0.7
        return HybridRetriever(dense, bm25, alpha=alpha)
    raise ValueError(f"Unknown retriever_type: {retriever_type!r}")


def create_reranker(reranker_type: str) -> BaseReranker:
    """Return the reranker for the given type ('cross_encoder' or 'cohere')."""
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    if reranker_type == "cohere":
        return CohereReranker()
    raise ValueError(f"Unknown reranker_type: {reranker_type!r}")


def create_llm(
    model: str = "gpt-4o-mini",
    cache: JSONCache | None = None,
) -> BaseLLM:
    """Return a LiteLLMClient with the given model and optional cache."""
    from src.generator import LiteLLMClient

    return LiteLLMClient(model=model, cache=cache)


def load_configs(config_dir: str) -> list[ExperimentConfig]:
    """Load and validate all YAML experiment configs from a directory.

    Files are processed in alphabetical order (naming convention 01_, 02_, ...
    controls execution order). Uses yaml.safe_load() — no arbitrary code execution.
    """
    import yaml
    from pathlib import Path

    configs: list[ExperimentConfig] = []
    for yaml_file in sorted(Path(config_dir).glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        configs.append(ExperimentConfig.model_validate(data))
    return configs
