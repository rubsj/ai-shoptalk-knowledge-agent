"""Tests for src/iteration_log.py — single-param diff detection, delta math, edge cases."""

from __future__ import annotations

import json

import pytest

from src.iteration_log import IterationEntry, build_iteration_log, save_iteration_log


def _make_result(
    exp_id: str = "exp-1",
    chunking: str = "recursive",
    embedding: str = "minilm",
    retriever: str = "dense",
    ndcg: float = 0.75,
    recall: float = 0.80,
    precision: float = 0.40,
    mrr: float = 0.70,
    hybrid_alpha: float | None = None,
    use_reranking: bool = False,
    reranker_type: str | None = None,
) -> dict:
    return {
        "experiment_id": exp_id,
        "config": {
            "chunking_strategy": chunking,
            "embedding_model": embedding,
            "retriever_type": retriever,
            "hybrid_alpha": hybrid_alpha,
            "use_reranking": use_reranking,
            "reranker_type": reranker_type,
        },
        "metrics": {
            "recall_at_5": recall,
            "precision_at_5": precision,
            "mrr": mrr,
            "ndcg_at_5": ndcg,
        },
    }


class TestBuildIterationLog:
    def test_empty_input(self):
        assert build_iteration_log([]) == []

    def test_single_result_no_pairs(self):
        results = [_make_result()]
        entries = build_iteration_log(results)
        assert entries == []

    def test_single_param_diff_detected(self):
        """Two configs differing only on embedding_model should produce an entry."""
        results = [
            _make_result("exp-a", embedding="minilm", ndcg=0.70),
            _make_result("exp-b", embedding="mpnet", ndcg=0.75),
        ]
        entries = build_iteration_log(results)
        assert len(entries) >= 1
        # Find the embedding_model entry
        em_entries = [e for e in entries if e.parameter_changed == "embedding_model"]
        assert len(em_entries) == 1
        entry = em_entries[0]
        assert entry.old_value == "minilm"
        assert entry.new_value == "mpnet"

    def test_delta_math_correct(self):
        results = [
            _make_result("exp-a", embedding="minilm", ndcg=0.60, recall=0.70, mrr=0.50),
            _make_result("exp-b", embedding="mpnet", ndcg=0.80, recall=0.90, mrr=0.70),
        ]
        entries = build_iteration_log(results)
        em_entries = [e for e in entries if e.parameter_changed == "embedding_model"]
        assert len(em_entries) == 1
        entry = em_entries[0]
        # "before" = lower NDCG (minilm), "after" = higher (mpnet)
        assert abs(entry.delta["ndcg_at_5"] - 0.20) < 1e-6
        assert abs(entry.delta["recall_at_5"] - 0.20) < 1e-6
        assert abs(entry.delta["mrr"] - 0.20) < 1e-6

    def test_two_param_diff_no_match(self):
        """Configs differing on TWO params should NOT pair for either param."""
        results = [
            _make_result("exp-a", chunking="recursive", embedding="minilm", ndcg=0.70),
            _make_result("exp-b", chunking="fixed", embedding="mpnet", ndcg=0.75),
        ]
        entries = build_iteration_log(results)
        # Should still find entries because each param is checked independently:
        # For chunking_strategy: they also differ on embedding, so no match
        # For embedding_model: they also differ on chunking, so no match
        assert len(entries) == 0

    def test_multiple_pairs_sorted_by_impact(self):
        results = [
            _make_result("exp-a", embedding="minilm", ndcg=0.70),
            _make_result("exp-b", embedding="mpnet", ndcg=0.72),
            _make_result("exp-c", embedding="openai", ndcg=0.90),
        ]
        entries = build_iteration_log(results)
        em_entries = [e for e in entries if e.parameter_changed == "embedding_model"]
        assert len(em_entries) >= 2
        # Sorted by absolute delta descending
        deltas = [abs(e.delta["ndcg_at_5"]) for e in em_entries]
        assert deltas == sorted(deltas, reverse=True)

    def test_iteration_ids_sequential(self):
        results = [
            _make_result("exp-a", embedding="minilm", ndcg=0.70),
            _make_result("exp-b", embedding="mpnet", ndcg=0.75),
            _make_result("exp-c", embedding="openai", ndcg=0.85),
        ]
        entries = build_iteration_log(results)
        ids = [e.iteration_id for e in entries]
        assert ids == list(range(1, len(ids) + 1))

    def test_reranking_diff(self):
        """use_reranking and reranker_type change together — iteration log captures both."""
        results = [
            _make_result("exp-a", use_reranking=False, reranker_type=None, ndcg=0.70),
            _make_result("exp-b", use_reranking=True, reranker_type="cross_encoder", ndcg=0.78),
        ]
        entries = build_iteration_log(results)
        # Both use_reranking AND reranker_type differ, so no single-param match
        # This is expected: reranking is a 2-param change (flag + type)
        rr_entries = [e for e in entries if e.parameter_changed in ("use_reranking", "reranker_type")]
        assert len(rr_entries) == 0

    def test_reranker_type_diff_same_flag(self):
        """Two reranked configs with different reranker_type produce an entry."""
        results = [
            _make_result("exp-a", use_reranking=True, reranker_type="cross_encoder", ndcg=0.70),
            _make_result("exp-b", use_reranking=True, reranker_type="cohere", ndcg=0.78),
        ]
        entries = build_iteration_log(results)
        rt_entries = [e for e in entries if e.parameter_changed == "reranker_type"]
        assert len(rt_entries) == 1


class TestSaveIterationLog:
    def test_saves_valid_json(self, tmp_path):
        entries = [
            IterationEntry(
                iteration_id=1,
                parameter_changed="embedding_model",
                old_value="minilm",
                new_value="mpnet",
                reason="test reason",
                experiment_id_before="exp-a",
                experiment_id_after="exp-b",
                metrics_before={"recall_at_5": 0.7, "precision_at_5": 0.3, "mrr": 0.5, "ndcg_at_5": 0.6},
                metrics_after={"recall_at_5": 0.8, "precision_at_5": 0.4, "mrr": 0.6, "ndcg_at_5": 0.7},
                delta={"recall_at_5": 0.1, "precision_at_5": 0.1, "mrr": 0.1, "ndcg_at_5": 0.1},
            )
        ]
        path = save_iteration_log(entries, str(tmp_path / "log.json"))
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["parameter_changed"] == "embedding_model"

    def test_creates_parent_dirs(self, tmp_path):
        path = save_iteration_log([], str(tmp_path / "sub" / "dir" / "log.json"))
        assert path.exists()
