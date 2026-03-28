"""Tests for reproducibility check in src/experiment_runner.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.schemas import (
    ExperimentConfig,
    ExperimentResult,
    PerformanceMetrics,
    RetrievalMetrics,
)


def _make_experiment_result(
    ndcg: float = 0.85,
    recall: float = 0.90,
    precision: float = 0.40,
    mrr: float = 0.80,
    exp_id: str = "test-exp",
) -> ExperimentResult:
    config = ExperimentConfig(
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="minilm",
        retriever_type="dense",
    )
    metrics = RetrievalMetrics(
        recall_at_5=recall,
        precision_at_5=precision,
        mrr=mrr,
        ndcg_at_5=ndcg,
    )
    performance = PerformanceMetrics(
        ingestion_time_seconds=2.0,
        avg_query_latency_ms=500.0,
        index_size_bytes=100000,
        peak_memory_mb=500.0,
        embedding_source="local",
        cost_estimate_usd=0.005,
    )
    return ExperimentResult(
        experiment_id=exp_id,
        config=config,
        metrics=metrics,
        performance=performance,
        query_results=[],
    )


class TestRunReproducibilityCheck:
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.load_ground_truth")
    def test_pass_when_metrics_identical(self, mock_gt, mock_embedder, mock_run):
        from src.experiment_runner import run_reproducibility_check

        run1 = _make_experiment_result(ndcg=0.85, recall=0.90, precision=0.40, mrr=0.80)
        # Run 2 returns identical metrics
        run2 = _make_experiment_result(ndcg=0.85, recall=0.90, precision=0.40, mrr=0.80)
        mock_run.return_value = run2
        mock_gt.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        result = run_reproducibility_check(
            results=[run1],
            documents=[],
            ground_truth_path="fake.json",
        )

        assert result["passed"] is True
        for metric_info in result["metrics"].values():
            assert metric_info["passed"] is True
            assert metric_info["delta"] == 0.0

    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.load_ground_truth")
    def test_pass_within_threshold(self, mock_gt, mock_embedder, mock_run):
        from src.experiment_runner import run_reproducibility_check

        run1 = _make_experiment_result(ndcg=0.85, recall=0.90)
        # Run 2 has small variance (< 5%)
        run2 = _make_experiment_result(ndcg=0.84, recall=0.89)
        mock_run.return_value = run2
        mock_gt.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        result = run_reproducibility_check(
            results=[run1],
            documents=[],
            ground_truth_path="fake.json",
        )

        assert result["passed"] is True

    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.load_ground_truth")
    def test_fail_when_exceeds_threshold(self, mock_gt, mock_embedder, mock_run):
        from src.experiment_runner import run_reproducibility_check

        run1 = _make_experiment_result(ndcg=0.85, recall=0.90)
        # Run 2 has large variance (> 5%)
        run2 = _make_experiment_result(ndcg=0.70, recall=0.70)
        mock_run.return_value = run2
        mock_gt.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        result = run_reproducibility_check(
            results=[run1],
            documents=[],
            ground_truth_path="fake.json",
        )

        assert result["passed"] is False

    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.load_ground_truth")
    def test_selects_best_by_ndcg(self, mock_gt, mock_embedder, mock_run):
        from src.experiment_runner import run_reproducibility_check

        low = _make_experiment_result(ndcg=0.60, exp_id="low")
        high = _make_experiment_result(ndcg=0.90, exp_id="high")
        mock_run.return_value = high
        mock_gt.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        result = run_reproducibility_check(
            results=[low, high],
            documents=[],
            ground_truth_path="fake.json",
        )

        assert result["best_experiment_id"] == "high"

    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.load_ground_truth")
    def test_custom_threshold(self, mock_gt, mock_embedder, mock_run):
        from src.experiment_runner import run_reproducibility_check

        run1 = _make_experiment_result(ndcg=0.85)
        run2 = _make_experiment_result(ndcg=0.84)  # ~1.2% variance
        mock_run.return_value = run2
        mock_gt.return_value = MagicMock()
        mock_embedder.return_value = MagicMock()

        # With strict 1% threshold, this should fail
        result = run_reproducibility_check(
            results=[run1],
            documents=[],
            ground_truth_path="fake.json",
            threshold=0.01,
        )
        assert result["passed"] is False

        # With lenient 5% threshold, this should pass
        result = run_reproducibility_check(
            results=[run1],
            documents=[],
            ground_truth_path="fake.json",
            threshold=0.05,
        )
        assert result["passed"] is True
