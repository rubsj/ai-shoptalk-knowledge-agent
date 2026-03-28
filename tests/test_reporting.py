"""Tests for src/reporting.py — comparison report generation."""

from __future__ import annotations

import pytest

from src.iteration_log import IterationEntry
from src.reporting import generate_comparison_report


def _make_result(
    exp_id: str = "exp-1",
    chunking: str = "recursive",
    embedding: str = "minilm",
    retriever: str = "dense",
    ndcg: float = 0.75,
    recall: float = 0.80,
    precision: float = 0.40,
    mrr: float = 0.70,
    use_reranking: bool = False,
    reranker_type: str | None = None,
    hybrid_alpha: float | None = None,
    latency: float = 1000.0,
    cost: float = 0.005,
    judge_scores: dict | None = None,
) -> dict:
    return {
        "experiment_id": exp_id,
        "config": {
            "chunking_strategy": chunking,
            "embedding_model": embedding if retriever != "bm25" else None,
            "retriever_type": retriever,
            "use_reranking": use_reranking,
            "reranker_type": reranker_type,
            "hybrid_alpha": hybrid_alpha,
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
            "window_size_tokens": None,
            "step_size_tokens": None,
            "breakpoint_threshold": 0.85,
            "min_chunk_size": 100,
        },
        "metrics": {
            "recall_at_5": recall,
            "precision_at_5": precision,
            "mrr": mrr,
            "ndcg_at_5": ndcg,
        },
        "performance": {
            "avg_query_latency_ms": latency,
            "ingestion_time_seconds": 2.0,
            "cost_estimate_usd": cost,
            "index_size_bytes": 100000,
            "peak_memory_mb": 500.0,
            "embedding_source": "local",
        },
        "judge_scores": judge_scores,
        "query_results": [],
    }


@pytest.fixture()
def sample_results():
    return [
        _make_result("exp-a", "recursive", "minilm", "dense", ndcg=0.70),
        _make_result("exp-b", "recursive", "mpnet", "dense", ndcg=0.75),
        _make_result("exp-c", "fixed", "minilm", "dense", ndcg=0.65),
        _make_result("exp-d", "recursive", "minilm", "hybrid", ndcg=0.72, hybrid_alpha=0.7),
        _make_result("exp-e", "recursive", "minilm", "bm25", ndcg=0.55),
        _make_result("exp-f", "recursive", "minilm", "dense", ndcg=0.78,
                     use_reranking=True, reranker_type="cross_encoder"),
        _make_result("exp-g", "heading_semantic", "openai", "dense", ndcg=0.85, cost=0.01),
    ]


@pytest.fixture()
def sample_iteration_log():
    return [
        IterationEntry(
            iteration_id=1,
            parameter_changed="embedding_model",
            old_value="minilm",
            new_value="openai",
            reason="OpenAI improved NDCG@5 by +0.15",
            experiment_id_before="exp-a",
            experiment_id_after="exp-g",
            metrics_before={"recall_at_5": 0.80, "precision_at_5": 0.40,
                            "mrr": 0.70, "ndcg_at_5": 0.70},
            metrics_after={"recall_at_5": 0.95, "precision_at_5": 0.45,
                           "mrr": 0.85, "ndcg_at_5": 0.85},
            delta={"recall_at_5": 0.15, "precision_at_5": 0.05,
                   "mrr": 0.15, "ndcg_at_5": 0.15},
        ),
    ]


class TestGenerateComparisonReport:
    def test_creates_markdown_file(self, sample_results, sample_iteration_log, tmp_path):
        path = generate_comparison_report(
            sample_results, sample_iteration_log, str(tmp_path / "report.md")
        )
        assert path.exists()
        assert path.suffix == ".md"
        assert path.stat().st_size > 0

    def test_contains_all_required_sections(self, sample_results, sample_iteration_log, tmp_path):
        path = generate_comparison_report(
            sample_results, sample_iteration_log, str(tmp_path / "report.md")
        )
        content = path.read_text()
        required_sections = [
            "## Summary",
            "## Q1",
            "## Q2",
            "## Q3",
            "## Q4",
            "## Best Configuration",
            "## Methodology",
            "## Iteration Log",
            "## Final Config Traceability",
            "## Judge Target Check",
            "## Self-Evaluation",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_contains_best_config_id(self, sample_results, sample_iteration_log, tmp_path):
        path = generate_comparison_report(
            sample_results, sample_iteration_log, str(tmp_path / "report.md")
        )
        content = path.read_text()
        # Best config is exp-g (ndcg=0.85)
        assert "exp-g" in content

    def test_contains_iteration_log_table(self, sample_results, sample_iteration_log, tmp_path):
        path = generate_comparison_report(
            sample_results, sample_iteration_log, str(tmp_path / "report.md")
        )
        content = path.read_text()
        assert "embedding_model" in content
        assert "minilm" in content
        assert "openai" in content

    def test_contains_traceability_table(self, sample_results, sample_iteration_log, tmp_path):
        path = generate_comparison_report(
            sample_results, sample_iteration_log, str(tmp_path / "report.md")
        )
        content = path.read_text()
        assert "Final Config Traceability" in content
        assert "heading_semantic" in content

    def test_no_iteration_log(self, sample_results, tmp_path):
        """Report generates without iteration log entries."""
        path = generate_comparison_report(
            sample_results, None, str(tmp_path / "report.md")
        )
        content = path.read_text()
        assert "No iteration log entries available" in content

    def test_with_judge_scores(self, tmp_path):
        judge = {
            "avg_relevance": 4.5,
            "avg_accuracy": 4.2,
            "avg_completeness": 4.0,
            "avg_conciseness": 4.8,
            "avg_citation_quality": 3.5,
            "overall_average": 4.2,
        }
        results = [_make_result("exp-j", judge_scores=judge)]
        path = generate_comparison_report(results, [], str(tmp_path / "report.md"))
        content = path.read_text()
        assert "4.20" in content  # overall average
        assert "PASS" in content  # 4.2 > 4.0 target

    def test_creates_parent_dirs(self, sample_results, tmp_path):
        path = generate_comparison_report(
            sample_results, [], str(tmp_path / "sub" / "dir" / "report.md")
        )
        assert path.exists()
