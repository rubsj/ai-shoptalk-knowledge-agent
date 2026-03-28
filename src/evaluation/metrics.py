"""Retrieval evaluation metrics — implemented from scratch (not sklearn).

Metrics:
    precision_at_k(retrieved_ids, relevant_ids, k) → float
    recall_at_k(retrieved_ids, relevant_ids, k) → float
    mrr(retrieved_ids, relevant_ids) → float          # Mean Reciprocal Rank
    ndcg_at_k(retrieved_ids, graded_relevance, k) → float
    compute_overlap_relevance(retrieved_chunks, gt_chunks) → (set[str], dict[str, int])

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import Chunk, GroundTruthChunk


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


def compute_overlap_relevance(
    retrieved_chunks: list[Chunk],
    gt_chunks: list[GroundTruthChunk],
    min_overlap_ratio: float = 0.3,
) -> tuple[set[str], dict[str, int]]:
    """Map retrieved chunks to GT relevance grades via character offset overlap.

    Solves the cross-chunker problem: ground truth was generated with one chunker
    (RecursiveChunker), but evaluation may use a different chunker → chunk IDs
    differ, making exact ID matching return 0.0 for all non-recursive configs.

    Matching strategy (per document):
        overlap = max(0, min(r_end, gt_end) - max(r_start, gt_start))
        ratio   = overlap / min(len_retrieved, len_gt)
        If ratio >= min_overlap_ratio → assign gt_chunk.relevance_grade

    Falls back to exact chunk ID matching when start_char/end_char are not set
    (e.g., legacy ground truth files or test fixtures without offset fields).

    Args:
        retrieved_chunks: Chunks returned by the retriever for one query.
        gt_chunks: GroundTruthChunk list for the same query.
        min_overlap_ratio: Minimum overlap fraction to count as a match (default 0.3).

    Returns:
        relevant_ids: Set of retrieved chunk IDs with grade >= 1.
        graded_relevance: Dict mapping retrieved chunk ID → best matching grade.
    """
    # Lazy import to avoid circular import at module load time
    from src.schemas import Chunk as _Chunk  # noqa: F401 (used for type check)

    graded: dict[str, int] = {}

    for r_chunk in retrieved_chunks:
        r_doc_id = r_chunk.metadata.document_id
        r_start = r_chunk.metadata.start_char
        r_end = r_chunk.metadata.end_char
        r_len = r_end - r_start

        best_grade = 0

        for gt in gt_chunks:
            # Exact ID fallback: legacy GT without offset fields
            if gt.start_char is None or gt.end_char is None or not gt.document_id:
                if r_chunk.id == gt.chunk_id:
                    best_grade = max(best_grade, gt.relevance_grade)
                continue

            if gt.document_id != r_doc_id:
                continue

            gt_len = gt.end_char - gt.start_char
            overlap = max(0, min(r_end, gt.end_char) - max(r_start, gt.start_char))
            denom = min(r_len, gt_len) if (r_len > 0 and gt_len > 0) else max(r_len, gt_len)
            if denom <= 0:
                continue

            ratio = overlap / denom
            if ratio >= min_overlap_ratio:
                best_grade = max(best_grade, gt.relevance_grade)

        if best_grade > 0:
            graded[r_chunk.id] = best_grade

    relevant_ids = {rid for rid, grade in graded.items() if grade >= 1}
    return relevant_ids, graded
