"""Evaluation framework: retrieval metrics, LLM-as-Judge, ground truth management.

Modules: metrics (IR metrics from scratch), judge (5-axis LLM judge),
ground_truth (loading/validation of curated queries).
"""

from src.evaluation.ground_truth import generate_ground_truth_candidates, load_ground_truth
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import compute_overlap_relevance, mrr, ndcg_at_k, precision_at_k, recall_at_k

__all__ = [
    "compute_overlap_relevance",
    "generate_ground_truth_candidates",
    "load_ground_truth",
    "LLMJudge",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
