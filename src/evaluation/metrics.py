"""Retrieval evaluation metrics — implemented from scratch (not sklearn).

Metrics:
    precision_at_k(retrieved_ids, relevant_ids, k) → float
    recall_at_k(retrieved_ids, relevant_ids, k) → float
    mrr(retrieved_ids, relevant_ids) → float          # Mean Reciprocal Rank
    ndcg_at_k(retrieved_ids, graded_relevance, k) → float

NDCG graded relevance scale (PRD Decision 3):
    3 = Gold chunk (directly answers the question)
    2 = Same section (contextually relevant)
    1 = Same document (topically related)
    0 = Irrelevant

Why from scratch: understanding NDCG's log₂ discount function is an interview
talking point. DCG@K = Σ(rel_i / log₂(i+1)). NDCG = DCG / ideal_DCG.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """% of top-K retrieved that are relevant. Returns 0.0 if k <= 0."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """% of all relevant chunks found in top-K. Returns 0.0 if relevant_ids is empty."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result. Returns 0.0 if none found."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _dcg(ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain for a ranked list up to position k.

    WHY log2(i+2): position i is 0-indexed, so rank = i+1, discount = log2(rank+1) = log2(i+2).
    This matches the standard IR formulation where DCG = Σ rel_i / log2(i+2).
    """
    total = 0.0
    for i, rid in enumerate(ids[:k]):
        rel = relevance.get(rid, 0)
        total += rel / math.log2(i + 2)
    return total


def ndcg_at_k(retrieved_ids: list[str], graded_relevance: dict[str, int], k: int) -> float:
    """NDCG@K with 4-level grading (0-3). Returns 0.0 if ideal DCG is 0 (all irrelevant).

    DCG@K  = Σ rel_i / log₂(i+2) over retrieved_ids[:k]
    iDCG@K = DCG of the perfect ordering (sort grades descending, take top-k)
    NDCG   = DCG / iDCG
    """
    if k <= 0:
        return 0.0

    dcg = _dcg(retrieved_ids, graded_relevance, k)

    # Ideal DCG: best possible ordering of all known relevant chunks
    ideal_grades = sorted(graded_relevance.values(), reverse=True)
    ideal_ids = [f"__ideal_{i}" for i in range(len(ideal_grades))]
    ideal_relevance = dict(zip(ideal_ids, ideal_grades))
    ideal_dcg = _dcg(ideal_ids, ideal_relevance, k)

    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg
