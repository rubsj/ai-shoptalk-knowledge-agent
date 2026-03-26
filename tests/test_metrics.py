"""Tests for src/evaluation/metrics.py — retrieval metrics from scratch.

Strategy:
  - Happy path with hand-computed expected values (verify we understand the math)
  - Edge cases: empty lists, k=0, no matches, perfect match
  - NDCG: perfect ranking = 1.0, worst ranking < 1.0, all-irrelevant = 0.0
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_partial(self):
        # 2 of top-3 are relevant → 2/3
        result = precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_k_smaller_than_retrieved(self):
        # only consider first 2: ["a","b"] — 1 relevant → 1/2
        result = precision_at_k(["a", "b", "c"], {"a", "x"}, k=2)
        assert abs(result - 0.5) < 1e-9

    def test_k_larger_than_retrieved(self):
        # only 2 items but k=5: 1 relevant in top-5 → 1/5
        result = precision_at_k(["a", "x"], {"a"}, k=5)
        assert abs(result - 1 / 5) < 1e-9

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self):
        assert precision_at_k(["a", "b"], set(), k=3) == 0.0

    def test_single_item_match(self):
        assert precision_at_k(["a"], {"a"}, k=1) == 1.0

    def test_single_item_no_match(self):
        assert precision_at_k(["b"], {"a"}, k=1) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_full_recall(self):
        # all 2 relevant in top-3 → 2/2 = 1.0
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self):
        # 1 of 2 relevant found in top-3 → 1/2
        result = recall_at_k(["a", "c", "d"], {"a", "b"}, k=3)
        assert abs(result - 0.5) < 1e-9

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], set(), k=3) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a", "b"}, k=5) == 0.0

    def test_k_limits_retrieval(self):
        # relevant "b" is at position 3, k=2 so not included → 0/1
        assert recall_at_k(["a", "c", "b"], {"b"}, k=2) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(["a", "b"], {"a"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_hit_at_rank_1(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_first_hit_at_rank_2(self):
        assert abs(mrr(["x", "a", "c"], {"a"}) - 0.5) < 1e-9

    def test_first_hit_at_rank_3(self):
        result = mrr(["x", "y", "a"], {"a"})
        assert abs(result - 1 / 3) < 1e-9

    def test_no_hit_returns_zero(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a", "b"], set()) == 0.0

    def test_multiple_relevant_uses_first(self):
        # "b" is at rank 2 but "a" is at rank 1 → MRR = 1.0
        assert mrr(["a", "b", "c"], {"a", "b"}) == 1.0

    def test_single_item_match(self):
        assert mrr(["a"], {"a"}) == 1.0

    def test_single_item_no_match(self):
        assert mrr(["b"], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCGAtK:
    def test_perfect_ranking_returns_1(self):
        # Retrieved in exact grade order: 3, 2, 1 → DCG = ideal_DCG
        graded = {"a": 3, "b": 2, "c": 1}
        result = ndcg_at_k(["a", "b", "c"], graded, k=3)
        assert abs(result - 1.0) < 1e-9

    def test_all_irrelevant_returns_zero(self):
        graded = {"a": 1, "b": 2}
        result = ndcg_at_k(["x", "y", "z"], graded, k=3)
        assert result == 0.0

    def test_empty_graded_relevance_returns_zero(self):
        # No relevant chunks defined → ideal_DCG = 0
        result = ndcg_at_k(["a", "b", "c"], {}, k=3)
        assert result == 0.0

    def test_k_zero_returns_zero(self):
        assert ndcg_at_k(["a", "b"], {"a": 3}, k=0) == 0.0

    def test_reversed_ranking_less_than_1(self):
        # Worst possible ordering: grade-1 first, grade-3 last
        graded = {"a": 3, "b": 1}
        perfect = ndcg_at_k(["a", "b"], graded, k=2)
        reversed_result = ndcg_at_k(["b", "a"], graded, k=2)
        assert perfect == 1.0
        assert reversed_result < 1.0

    def test_hand_computed_example(self):
        # Retrieved: a(grade=3), b(grade=0), c(grade=2), d(grade=1)
        # DCG = 3/log2(2) + 0/log2(3) + 2/log2(4) + 1/log2(5)
        #     = 3/1 + 0 + 2/2 + 1/log2(5)
        #     = 3 + 0 + 1 + 1/2.3219...
        # ideal order: 3, 2, 1, 0
        # iDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
        #      = 3 + 2/1.585 + 0.5 + 0
        graded = {"a": 3, "b": 0, "c": 2, "d": 1}
        retrieved = ["a", "b", "c", "d"]
        dcg = 3 / math.log2(2) + 0 / math.log2(3) + 2 / math.log2(4) + 1 / math.log2(5)
        ideal_dcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4) + 0 / math.log2(5)
        expected = dcg / ideal_dcg
        result = ndcg_at_k(retrieved, graded, k=4)
        assert abs(result - expected) < 1e-9

    def test_single_gold_chunk_found_first(self):
        # One gold chunk at rank 1 → NDCG = 1.0
        result = ndcg_at_k(["gold"], {"gold": 3}, k=1)
        assert abs(result - 1.0) < 1e-9

    def test_single_gold_chunk_not_found(self):
        result = ndcg_at_k(["x"], {"gold": 3}, k=1)
        assert result == 0.0

    def test_k_truncates_ranking(self):
        # With k=1: only first item matters
        graded = {"a": 1, "b": 3}
        # Retrieving b first with k=1: DCG = 3/log2(2), iDCG = 3/log2(2) → 1.0
        result_perfect = ndcg_at_k(["b", "a"], graded, k=1)
        # Retrieving a first with k=1: DCG = 1/log2(2), iDCG = 3/log2(2) → 1/3
        result_suboptimal = ndcg_at_k(["a", "b"], graded, k=1)
        assert abs(result_perfect - 1.0) < 1e-9
        assert abs(result_suboptimal - 1 / 3) < 1e-9

    def test_empty_retrieved_returns_zero(self):
        result = ndcg_at_k([], {"a": 3}, k=5)
        assert result == 0.0

    def test_grade_0_in_graded_dict_counts_as_irrelevant(self):
        # Chunk "x" has grade 0 → iDCG based on grade-0 is 0 → NDCG = 0
        result = ndcg_at_k(["x"], {"x": 0}, k=1)
        assert result == 0.0


# ---------------------------------------------------------------------------
# compute_overlap_relevance
# ---------------------------------------------------------------------------


from src.evaluation.metrics import compute_overlap_relevance
from src.schemas import Chunk, ChunkMetadata, GroundTruthChunk


def _make_chunk(chunk_id: str, doc_id: str, start: int, end: int) -> Chunk:
    """Helper: minimal Chunk with specified char offsets."""
    return Chunk(
        id=chunk_id,
        content="x" * (end - start),
        metadata=ChunkMetadata(
            document_id=doc_id,
            source="doc.pdf",
            page_number=0,
            start_char=start,
            end_char=end,
            chunk_index=0,
        ),
    )


def _gt(chunk_id: str, doc_id: str, start: int, end: int, grade: int) -> GroundTruthChunk:
    """Helper: GroundTruthChunk with all offset fields set."""
    return GroundTruthChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        start_char=start,
        end_char=end,
        relevance_grade=grade,
    )


class TestComputeOverlapRelevance:
    def test_exact_overlap_returns_gt_grade(self):
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc1", 0, 100, 3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert "r1" in rel_ids
        assert graded["r1"] == 3

    def test_partial_overlap_above_threshold_matches(self):
        # retrieved [0, 100], GT [50, 150] → overlap=50, min_len=100 → ratio=0.5 ≥ 0.3
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc1", 50, 150, 2)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert "r1" in rel_ids
        assert graded["r1"] == 2

    def test_partial_overlap_below_threshold_no_match(self):
        # retrieved [0, 100], GT [80, 200] → overlap=20, min_len=100 → ratio=0.2 < 0.3
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc1", 80, 200, 3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert "r1" not in rel_ids
        assert graded == {}

    def test_no_overlap_no_match(self):
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc1", 200, 300, 3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert rel_ids == set()
        assert graded == {}

    def test_different_document_not_matched(self):
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc2", 0, 100, 3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert rel_ids == set()

    def test_grade_0_excluded_from_relevant_ids(self):
        chunk = _make_chunk("r1", "doc1", 0, 100)
        gt = _gt("g1", "doc1", 0, 100, 0)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert rel_ids == set()

    def test_multiple_gt_takes_max_grade(self):
        chunk = _make_chunk("r1", "doc1", 0, 200)
        gt1 = _gt("g1", "doc1", 0, 100, 1)
        gt2 = _gt("g2", "doc1", 100, 200, 3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt1, gt2])
        assert graded["r1"] == 3

    def test_fallback_to_exact_id_when_offsets_missing(self):
        chunk = _make_chunk("chunk-abc", "doc1", 0, 100)
        gt = GroundTruthChunk(chunk_id="chunk-abc", relevance_grade=2)  # no offset fields
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert "chunk-abc" in rel_ids
        assert graded["chunk-abc"] == 2

    def test_fallback_id_mismatch_no_match(self):
        chunk = _make_chunk("chunk-xyz", "doc1", 0, 100)
        gt = GroundTruthChunk(chunk_id="chunk-abc", relevance_grade=3)
        rel_ids, graded = compute_overlap_relevance([chunk], [gt])
        assert rel_ids == set()

    def test_empty_retrieved_returns_empty(self):
        gt = _gt("g1", "doc1", 0, 100, 3)
        rel_ids, graded = compute_overlap_relevance([], [gt])
        assert rel_ids == set()
        assert graded == {}

    def test_empty_gt_returns_empty(self):
        chunk = _make_chunk("r1", "doc1", 0, 100)
        rel_ids, graded = compute_overlap_relevance([chunk], [])
        assert rel_ids == set()
        assert graded == {}

    def test_multiple_retrieved_chunks_scored_independently(self):
        chunk_a = _make_chunk("ra", "doc1", 0, 100)
        chunk_b = _make_chunk("rb", "doc1", 500, 600)
        gt = _gt("g1", "doc1", 0, 100, 3)
        rel_ids, graded = compute_overlap_relevance([chunk_a, chunk_b], [gt])
        assert "ra" in rel_ids
        assert "rb" not in rel_ids
