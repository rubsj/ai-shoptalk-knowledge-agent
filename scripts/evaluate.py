"""CLI: run experiment grid — executes all configs, saves results JSON per config.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --no-judge
    python scripts/evaluate.py --configs experiments/configs/ --ground-truth data/ground_truth.json
    python scripts/evaluate.py --reproducibility-check

Deliverable D7 (PRD Section 7b): 46 JSON result files, each with config + metrics.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_runner import run_experiment_grid  # noqa: E402
from src.extraction import extract_all_pdfs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--configs",
    default="experiments/configs",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing YAML experiment configs (default: experiments/configs)",
)
@click.option(
    "--ground-truth",
    default="data/ground_truth.json",
    type=click.Path(exists=True, path_type=Path),
    help="Path to ground truth JSON (default: data/ground_truth.json)",
)
@click.option(
    "--output",
    default="results/experiments",
    type=click.Path(path_type=Path),
    help="Output directory for result JSONs (default: results/experiments)",
)
@click.option(
    "--pdfs",
    default="data/pdfs",
    type=click.Path(exists=True, path_type=Path),
    help="PDF directory (default: data/pdfs)",
)
@click.option(
    "--no-judge",
    is_flag=True,
    help="Skip LLM-as-Judge scoring (retrieval metrics only — faster, cheaper)",
)
@click.option(
    "--reproducibility-check",
    is_flag=True,
    help="After grid, re-run best config twice and compare metrics",
)
def main(
    configs: Path,
    ground_truth: Path,
    output: Path,
    pdfs: Path,
    no_judge: bool,
    reproducibility_check: bool,
) -> None:
    """Run RAG experiment grid over all YAML configs."""
    # Load documents from cache
    logger.info("Loading documents from %s...", pdfs)
    documents = extract_all_pdfs(str(pdfs))
    logger.info(
        "Loaded %d documents (%d chars total)",
        len(documents),
        sum(len(d.content) for d in documents),
    )

    # Run grid
    run_judge = not no_judge
    logger.info("Running experiment grid (judge=%s)...", run_judge)
    start = time.monotonic()

    results = run_experiment_grid(
        config_dir=str(configs),
        ground_truth_path=str(ground_truth),
        output_dir=str(output),
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
        print(
            f"  {label:<45} {r.metrics.ndcg_at_5:>7.3f} {r.metrics.recall_at_5:>9.3f} "
            f"{r.metrics.mrr:>6.3f} {r.performance.avg_query_latency_ms:>7.0f}ms"
        )

    # Best config
    best = max(results, key=lambda x: x.metrics.ndcg_at_5)
    bc = best.config
    print(f"\n  Best config: {bc.chunking_strategy}_{bc.embedding_model}_{bc.retriever_type}")
    print(
        f"  NDCG@5={best.metrics.ndcg_at_5:.3f}  Recall@5={best.metrics.recall_at_5:.3f}  "
        f"MRR={best.metrics.mrr:.3f}"
    )
    if best.judge_scores:
        js = best.judge_scores
        print(
            f"  Judge: relevance={js.avg_relevance:.1f} accuracy={js.avg_accuracy:.1f} "
            f"completeness={js.avg_completeness:.1f} conciseness={js.avg_conciseness:.1f} "
            f"citation={js.avg_citation_quality:.1f} overall={js.overall_average:.1f}"
        )

    # Total cost
    total_cost = sum(r.performance.cost_estimate_usd for r in results)
    print(f"\n  Total estimated cost: ${total_cost:.4f}")
    print(f"  Wall time: {elapsed / 60:.1f} minutes")
    print(f"  Results saved to: {output}/")
    print(f"{'=' * 80}")

    # Reproducibility check
    if reproducibility_check:
        logger.info("Running reproducibility check on best config...")
        from src.experiment_runner import run_reproducibility_check

        check = run_reproducibility_check(
            results=results,
            documents=documents,
            ground_truth_path=str(ground_truth),
        )

        print("\n  REPRODUCIBILITY CHECK")
        print(f"  {'Metric':<15} {'Run 1':>8} {'Run 2':>8} {'Delta':>8} {'Status':>8}")
        metric_labels = {
            "ndcg_at_5": "NDCG@5",
            "recall_at_5": "Recall@5",
            "precision_at_5": "Precision@5",
            "mrr": "MRR",
        }
        for metric, info in check["metrics"].items():
            status = "PASS" if info["passed"] else "FAIL"
            print(
                f"  {metric_labels[metric]:<15} {info['run1']:>8.4f} {info['run2']:>8.4f} "
                f"{info['delta']:>8.4f} {status:>8}"
            )
        print(f"\n  Reproducibility: {'PASSED' if check['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
