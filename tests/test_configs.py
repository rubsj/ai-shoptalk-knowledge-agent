"""Tests for experiments/configs/*.yaml — experiment grid structure and Pydantic validation."""

from __future__ import annotations

from src.factories import load_configs
from src.schemas import ExperimentConfig


CONFIGS_DIR = "experiments/configs"


def _all_configs() -> list[ExperimentConfig]:
    return load_configs(CONFIGS_DIR)


class TestConfigCount:
    def test_at_least_35_configs(self) -> None:
        assert len(_all_configs()) >= 35

    def test_exactly_52_configs(self) -> None:
        assert len(_all_configs()) == 52


class TestConfigValidity:
    def test_all_are_experiment_config(self) -> None:
        for config in _all_configs():
            assert isinstance(config, ExperimentConfig)

    def test_all_have_valid_chunking_strategy(self) -> None:
        valid = {"fixed", "recursive", "sliding_window", "heading_semantic", "embedding_semantic"}
        for config in _all_configs():
            assert config.chunking_strategy in valid

    def test_all_have_valid_retriever_type(self) -> None:
        valid = {"dense", "bm25", "hybrid"}
        for config in _all_configs():
            assert config.retriever_type in valid


class TestBM25Configs:
    def test_bm25_configs_have_no_embedding_model(self) -> None:
        bm25_configs = [c for c in _all_configs() if c.retriever_type == "bm25"]
        assert len(bm25_configs) == 5
        for config in bm25_configs:
            assert config.embedding_model is None

    def test_bm25_covers_all_chunking_strategies(self) -> None:
        bm25_configs = [c for c in _all_configs() if c.retriever_type == "bm25"]
        strategies = {c.chunking_strategy for c in bm25_configs}
        assert strategies == {"fixed", "recursive", "sliding_window", "heading_semantic", "embedding_semantic"}


class TestHybridConfigs:
    def test_hybrid_configs_have_alpha_set(self) -> None:
        hybrid_configs = [c for c in _all_configs() if c.retriever_type == "hybrid"]
        # 15 base + 3 alpha sweep + 2 cross_encoder rerank + 2 cohere rerank = 22, + 3 ollama = 25
        assert len(hybrid_configs) == 25
        for config in hybrid_configs:
            assert config.hybrid_alpha is not None

    def test_alpha_sweep_values(self) -> None:
        hybrid_configs = [c for c in _all_configs() if c.retriever_type == "hybrid"]
        alphas = sorted({c.hybrid_alpha for c in hybrid_configs})
        assert alphas == [0.3, 0.5, 0.7, 0.9]

    def test_hybrid_covers_all_embedders(self) -> None:
        hybrid_configs = [c for c in _all_configs() if c.retriever_type == "hybrid"]
        embedders = {c.embedding_model for c in hybrid_configs}
        assert embedders == {"minilm", "mpnet", "openai", "ollama_nomic"}


class TestSlidingWindowConfigs:
    def test_sliding_window_configs_have_window_params(self) -> None:
        sliding = [c for c in _all_configs() if c.chunking_strategy == "sliding_window"]
        assert len(sliding) == 9
        for config in sliding:
            assert config.window_size_tokens == 128
            assert config.step_size_tokens == 64

    def test_non_sliding_configs_have_no_window_params(self) -> None:
        non_sliding = [c for c in _all_configs() if c.chunking_strategy != "sliding_window"]
        for config in non_sliding:
            assert config.window_size_tokens is None
            assert config.step_size_tokens is None


class TestOpenAIConfigs:
    def test_openai_configs_count(self) -> None:
        openai_configs = [c for c in _all_configs() if c.embedding_model == "openai"]
        assert len(openai_configs) == 10

    def test_openai_configs_cover_dense_and_hybrid(self) -> None:
        openai_configs = [c for c in _all_configs() if c.embedding_model == "openai"]
        retriever_types = {c.retriever_type for c in openai_configs}
        assert retriever_types == {"dense", "hybrid"}


class TestCommonDefaults:
    def test_chunk_size_is_512(self) -> None:
        for config in _all_configs():
            assert config.chunk_size == 512

    def test_chunk_overlap_is_50(self) -> None:
        for config in _all_configs():
            assert config.chunk_overlap == 50

    def test_top_k_is_5(self) -> None:
        for config in _all_configs():
            assert config.top_k == 5

    def test_reranking_configs_have_reranker_type(self) -> None:
        rerank_configs = [c for c in _all_configs() if c.use_reranking]
        assert len(rerank_configs) == 8
        types = {c.reranker_type for c in rerank_configs}
        assert types == {"cross_encoder", "cohere"}

    def test_non_reranking_configs_have_no_reranker(self) -> None:
        non_rerank = [c for c in _all_configs() if not c.use_reranking]
        for config in non_rerank:
            assert config.reranker_type is None
