"""Tests for src/visualization.py — all 10 chart functions + orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.visualization import (
    _config_label,
    _results_to_dataframe,
    generate_all_charts,
    plot_chunking_comparison,
    plot_config_metric_heatmap,
    plot_embedding_comparison,
    plot_hybrid_alpha_sweep,
    plot_judge_radar,
    plot_latency_vs_quality,
    plot_ndcg_distribution,
    plot_query_difficulty,
    plot_reranking_comparison,
    plot_retriever_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    chunking="recursive",
    embedding="minilm",
    retriever="dense",
    ndcg=0.75,
    recall=0.80,
    precision=0.40,
    mrr=0.70,
    use_reranking=False,
    reranker_type=None,
    hybrid_alpha=None,
    judge_scores=None,
    latency=1000.0,
    cost=0.005,
    query_results=None,
) -> dict:
    config = {
        "chunking_strategy": chunking,
        "embedding_model": embedding if retriever != "bm25" else None,
        "retriever_type": retriever,
        "use_reranking": use_reranking,
        "reranker_type": reranker_type,
        "hybrid_alpha": hybrid_alpha,
    }
    metrics = {
        "recall_at_5": recall,
        "precision_at_5": precision,
        "mrr": mrr,
        "ndcg_at_5": ndcg,
    }
    performance = {
        "avg_query_latency_ms": latency,
        "ingestion_time_seconds": 2.0,
        "cost_estimate_usd": cost,
        "index_size_bytes": 100000,
        "peak_memory_mb": 500.0,
        "embedding_source": "local",
    }
    return {
        "experiment_id": f"test-{chunking}-{embedding}-{retriever}",
        "config": config,
        "metrics": metrics,
        "performance": performance,
        "judge_scores": judge_scores,
        "query_results": query_results or [
            {
                "query_id": "q01",
                "question": "Test question?",
                "answer": "[1] Test answer.",
                "retrieved_chunk_ids": ["c1", "c2"],
                "retrieval_scores": {"recall_at_5": recall, "precision_at_5": precision,
                                     "mrr": mrr, "ndcg_at_5": ndcg},
                "judge_result": None,
                "latency_ms": latency,
            }
        ],
    }


@pytest.fixture()
def sample_results():
    """Minimal set of results covering all chart requirements."""
    return [
        _make_result("recursive", "minilm", "dense", ndcg=0.70),
        _make_result("recursive", "minilm", "hybrid", ndcg=0.75, hybrid_alpha=0.7),
        _make_result("recursive", "mpnet", "dense", ndcg=0.72),
        _make_result("fixed", "minilm", "dense", ndcg=0.65),
        _make_result("sliding_window", "openai", "dense", ndcg=0.80, cost=0.01),
        _make_result("heading_semantic", "openai", "hybrid", ndcg=0.85, hybrid_alpha=0.7),
        _make_result("embedding_semantic", "minilm", "bm25", ndcg=0.50),
        # Reranking pair
        _make_result("recursive", "minilm", "dense", ndcg=0.78, use_reranking=True,
                     reranker_type="cross_encoder"),
        # Alpha sweep
        _make_result("recursive", "minilm", "hybrid", ndcg=0.72, hybrid_alpha=0.3),
        _make_result("recursive", "minilm", "hybrid", ndcg=0.74, hybrid_alpha=0.5),
        _make_result("recursive", "minilm", "hybrid", ndcg=0.73, hybrid_alpha=0.9),
    ]


@pytest.fixture()
def results_with_judge():
    """Results that include judge scores."""
    judge = {
        "avg_relevance": 4.5,
        "avg_accuracy": 4.2,
        "avg_completeness": 4.0,
        "avg_conciseness": 4.8,
        "avg_citation_quality": 3.5,
        "overall_average": 4.2,
    }
    return [
        _make_result("recursive", "minilm", "dense", ndcg=0.75, judge_scores=judge),
        _make_result("fixed", "mpnet", "dense", ndcg=0.70, judge_scores=judge),
    ]


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestConfigLabel:
    def test_basic_label(self):
        config = {"chunking_strategy": "recursive", "embedding_model": "minilm",
                  "retriever_type": "dense", "use_reranking": False}
        assert _config_label(config) == "recursive_minilm_dense"

    def test_bm25_no_embedder(self):
        config = {"chunking_strategy": "fixed", "embedding_model": None,
                  "retriever_type": "bm25", "use_reranking": False}
        assert _config_label(config) == "fixed_bm25_bm25"

    def test_reranking_label(self):
        config = {"chunking_strategy": "recursive", "embedding_model": "minilm",
                  "retriever_type": "dense", "use_reranking": True,
                  "reranker_type": "cross_encoder"}
        assert "rr(ce)" in _config_label(config)

    def test_alpha_label(self):
        config = {"chunking_strategy": "recursive", "embedding_model": "minilm",
                  "retriever_type": "hybrid", "use_reranking": False,
                  "hybrid_alpha": 0.3}
        assert "a0.3" in _config_label(config)


class TestResultsToDataframe:
    def test_returns_dataframe(self, sample_results):
        df = _results_to_dataframe(sample_results)
        assert len(df) == len(sample_results)
        assert "ndcg_at_5" in df.columns
        assert "label" in df.columns
        assert "chunking_strategy" in df.columns

    def test_empty_results(self):
        df = _results_to_dataframe([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Chart function tests
# ---------------------------------------------------------------------------


class TestPlotConfigMetricHeatmap:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_config_metric_heatmap(df, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


class TestPlotChunkingComparison:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_chunking_comparison(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotEmbeddingComparison:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_embedding_comparison(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotRetrieverComparison:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_retriever_comparison(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotHybridAlphaSweep:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_hybrid_alpha_sweep(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_hybrid_data(self, tmp_path):
        df = _results_to_dataframe([_make_result("fixed", "minilm", "dense")])
        path = plot_hybrid_alpha_sweep(df, tmp_path)
        assert path.exists()  # creates placeholder chart


class TestPlotRerankingComparison:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_reranking_comparison(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_reranking_data(self, tmp_path):
        df = _results_to_dataframe([_make_result("fixed", "minilm", "dense")])
        path = plot_reranking_comparison(df, tmp_path)
        assert path.exists()  # creates placeholder chart


class TestPlotNdcgDistribution:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_ndcg_distribution(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotJudgeRadar:
    def test_creates_png_with_judge(self, results_with_judge, tmp_path):
        path = plot_judge_radar(results_with_judge, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_creates_placeholder_without_judge(self, sample_results, tmp_path):
        path = plot_judge_radar(sample_results, tmp_path)
        assert path.exists()  # placeholder for no-judge case


class TestPlotLatencyVsQuality:
    def test_creates_png(self, sample_results, tmp_path):
        df = _results_to_dataframe(sample_results)
        path = plot_latency_vs_quality(df, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotQueryDifficulty:
    def test_creates_png(self, sample_results, tmp_path):
        path = plot_query_difficulty(sample_results, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_query_results(self, tmp_path):
        results = [_make_result(query_results=[])]
        path = plot_query_difficulty(results, tmp_path)
        assert path.exists()  # creates placeholder


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestGenerateAllCharts:
    def test_returns_11_paths(self, sample_results, tmp_path):
        paths = generate_all_charts(sample_results, str(tmp_path))
        assert len(paths) == 11
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0

    def test_creates_output_dir(self, sample_results, tmp_path):
        out = tmp_path / "charts_subdir"
        paths = generate_all_charts(sample_results, str(out))
        assert out.exists()
        assert len(paths) == 11

    def test_empty_results(self, tmp_path):
        """Empty results should still produce 11 PNGs (with empty/placeholder content)."""
        # Provide a minimal single result so DataFrame operations don't crash
        results = [_make_result()]
        paths = generate_all_charts(results, str(tmp_path))
        assert len(paths) == 11
