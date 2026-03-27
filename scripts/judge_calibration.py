"""Judge calibration — select 5 diverse query-answer pairs for human scoring.

Picks pairs from different configs and different source papers so the calibration
covers the range of system behavior. Writes to results/judge_calibration_input.json.

Usage:
    python scripts/judge_calibration.py
    python scripts/judge_calibration.py --results results/experiments/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _pick_diverse_pairs(results: list[dict], n: int = 5) -> list[dict]:
    """Select n diverse query-answer pairs across configs and papers."""
    # Strategy: pick from configs spread across the NDCG@5 range,
    # and prefer different query_ids to cover different papers.
    sorted_results = sorted(results, key=lambda r: r["metrics"]["ndcg_at_5"], reverse=True)

    # Pick configs at evenly-spaced positions in the ranking
    step = max(1, len(sorted_results) // n)
    selected_configs = [sorted_results[i * step] for i in range(min(n, len(sorted_results)))]

    pairs = []
    used_query_ids: set[str] = set()

    for config_result in selected_configs:
        query_results = config_result.get("query_results", [])
        if not query_results:
            continue

        # Pick a query we haven't used yet
        chosen = None
        for qr in query_results:
            if qr["query_id"] not in used_query_ids:
                chosen = qr
                break
        if chosen is None:
            chosen = query_results[0]

        used_query_ids.add(chosen["query_id"])

        config = config_result["config"]
        label = (f"{config['chunking_strategy']}_{config.get('embedding_model') or 'bm25'}"
                 f"_{config['retriever_type']}")

        pair = {
            "pair_id": len(pairs) + 1,
            "config_label": label,
            "experiment_id": config_result["experiment_id"],
            "query_id": chosen["query_id"],
            "question": chosen["question"],
            "answer": chosen["answer"],
            "retrieved_chunk_ids": chosen["retrieved_chunk_ids"],
            "context_chunks": [
                {"chunk_id": cid, "rank": i + 1}
                for i, cid in enumerate(chosen["retrieved_chunk_ids"])
            ],
            "llm_judge_scores": chosen.get("judge_result"),
            "human_scores": {
                "relevance": None,
                "accuracy": None,
                "completeness": None,
                "conciseness": None,
                "citation_quality": None,
            },
        }
        pairs.append(pair)

        if len(pairs) >= n:
            break

    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Select pairs for judge calibration")
    parser.add_argument(
        "--results",
        default="results/experiments/summary.json",
        help="Path to experiment results JSON",
    )
    parser.add_argument(
        "--output",
        default="results/judge_calibration_input.json",
        help="Output path for calibration pairs",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of pairs to select (default: 5)",
    )
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"ERROR: Results file not found: {args.results}", file=sys.stderr)
        return 1

    results = json.load(open(args.results))
    pairs = _pick_diverse_pairs(results, n=args.n)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(pairs, indent=2))

    print(f"""
=== JUDGE CALIBRATION — DEVELOPER REQUIRED ===
{len(pairs)} query-answer pairs saved to {args.output}

For each pair, read the answer + context chunks and score on 5 axes (1-5):
- Relevance: 1=off-topic → 5=directly answers
- Accuracy: 1=major errors → 5=every claim verifiable
- Completeness: 1=fragment → 5=comprehensive
- Conciseness: 1=verbose → 5=focused
- Citation Quality: 1=no citations → 5=every claim cited

Add your scores as "human_scores" to each entry in the JSON file.
Reply 'continue' when done.

Selected pairs:""")
    for p in pairs:
        judge_info = ""
        if p["llm_judge_scores"]:
            js = p["llm_judge_scores"]
            avg = sum(js.values()) / len(js) if js else 0
            judge_info = f" (LLM judge avg: {avg:.1f})"
        print(f"  {p['pair_id']}. [{p['config_label']}] {p['query_id']}: "
              f"{p['question'][:60]}...{judge_info}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
