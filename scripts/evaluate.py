"""CLI: run experiment grid — executes all configs, saves results JSON per config.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --no-judge
    python scripts/evaluate.py --configs experiments/configs/ --ground-truth data/ground_truth.json
    python scripts/evaluate.py --reproducibility-check

Deliverable D7 (PRD Section 7b): 46 JSON result files, each with config + metrics.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_runner import run_experiment_grid  # noqa: E402
from src.extraction import extract_all_pdfs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG experiment grid")
    parser.add_argument(
        "--configs",
        default="experiments/configs",
        help="Directory containing YAML experiment configs (default: experiments/configs)",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth.json",
        help="Path to ground truth JSON (default: data/ground_truth.json)",
    )
    parser.add_argument(
        "--output",
        default="results/experiments",
        help="Output directory for result JSONs (default: results/experiments)",
    )
    parser.add_argument(
        "--pdfs",
        default="data/pdfs",
        help="PDF directory (default: data/pdfs)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-Judge scoring (retrieval metrics only — faster, cheaper)",
    )
    parser.add_argument(
        "--reproducibility-check",
        action="store_true",
        help="After grid, re-run best config twice and compare metrics",
    )
    args = parser.parse_args()

    # Validate paths
    if not Path(args.configs).exists():
        print(f"ERROR: Config directory not found: {args.configs}", file=sys.stderr)
        return 1
    if not Path(args.ground_truth).exists():
        print(f"ERROR: Ground truth not found: {args.ground_truth}", file=sys.stderr)
        return 1

    # Load documents from cache
    logger.info("Loading documents from %s...", args.pdfs)
    documents = extract_all_pdfs(args.pdfs)
    logger.info("Loaded %d documents (%d chars total)",
                len(documents), sum(len(d.content) for d in documents))

    # Run grid
    run_judge = not args.no_judge
    logger.info("Running experiment grid (judge=%s)...", run_judge)
    start = time.monotonic()

    results = run_experiment_grid(
        config_dir=args.configs,
        ground_truth_path=args.ground_truth,
        output_dir=args.output,
        documents=documents,
        run_judge=run_judge,
    )

    elapsed = time.monotonic() - start
    logger.info("Grid complete: %d results in %.1f minutes", len(results), elapsed / 60)

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"  EXPERIMENT GRID RESULTS — {len(results)} configs")
    print(f"{'=' * 80}")
    print(f"  {'Config':<45} {'NDCG@5':>7} {'Recall@5':>9} {'MRR':>6} {'Latency':>8}")
    print(f"  {'-' * 45} {'-' * 7} {'-' * 9} {'-' * 6} {'-' * 8}")

    for r in sorted(results, key=lambda x: x.metrics.ndcg_at_5, reverse=True):
        c = r.config
        label = f"{c.chunking_strategy}_{c.embedding_model or 'bm25'}_{c.retriever_type}"
        if c.use_reranking:
            label += f"_rerank({c.reranker_type})"
        if c.hybrid_alpha is not None and c.hybrid_alpha != 0.7:
            label += f"_a{c.hybrid_alpha}"
        print(f"  {label:<45} {r.metrics.ndcg_at_5:>7.3f} {r.metrics.recall_at_5:>9.3f} "
              f"{r.metrics.mrr:>6.3f} {r.performance.avg_query_latency_ms:>7.0f}ms")

    # Best config
    best = max(results, key=lambda x: x.metrics.ndcg_at_5)
    bc = best.config
    print(f"\n  Best config: {bc.chunking_strategy}_{bc.embedding_model}_{bc.retriever_type}")
    print(f"  NDCG@5={best.metrics.ndcg_at_5:.3f}  Recall@5={best.metrics.recall_at_5:.3f}  "
          f"MRR={best.metrics.mrr:.3f}")
    if best.judge_scores:
        js = best.judge_scores
        print(f"  Judge: relevance={js.avg_relevance:.1f} accuracy={js.avg_accuracy:.1f} "
              f"completeness={js.avg_completeness:.1f} conciseness={js.avg_conciseness:.1f} "
              f"citation={js.avg_citation_quality:.1f} overall={js.overall_average:.1f}")

    # Total cost
    total_cost = sum(r.performance.cost_estimate_usd for r in results)
    print(f"\n  Total estimated cost: ${total_cost:.4f}")
    print(f"  Wall time: {elapsed / 60:.1f} minutes")
    print(f"  Results saved to: {args.output}/")
    print(f"{'=' * 80}")

    # Reproducibility check
    if args.reproducibility_check:
        logger.info("Running reproducibility check on best config...")
        from src.experiment_runner import _run_single_config
        from src.evaluation.ground_truth import load_ground_truth
        from src.factories import create_embedder
        from src.cache import JSONCache

        gt = load_ground_truth(args.ground_truth)
        embedder = create_embedder(bc.embedding_model) if bc.embedding_model else None
        cache = JSONCache(str(Path(args.output) / "llm_cache"))

        run2 = _run_single_config(bc, embedder, documents, gt, judge=None, cache=cache)

        print(f"\n  REPRODUCIBILITY CHECK")
        print(f"  {'Metric':<15} {'Run 1':>8} {'Run 2':>8} {'Delta':>8} {'Status':>8}")
        metrics_pairs = [
            ("NDCG@5", best.metrics.ndcg_at_5, run2.metrics.ndcg_at_5),
            ("Recall@5", best.metrics.recall_at_5, run2.metrics.recall_at_5),
            ("Precision@5", best.metrics.precision_at_5, run2.metrics.precision_at_5),
            ("MRR", best.metrics.mrr, run2.metrics.mrr),
        ]
        all_pass = True
        for name, v1, v2 in metrics_pairs:
            delta = abs(v1 - v2)
            threshold = max(v1, 0.001) * 0.05
            status = "PASS" if delta < threshold else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {name:<15} {v1:>8.4f} {v2:>8.4f} {delta:>8.4f} {status:>8}")
        print(f"\n  Reproducibility: {'PASSED' if all_pass else 'FAILED'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
