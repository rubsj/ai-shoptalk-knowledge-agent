# ADR-003: Hybrid Retrieval with Min-Max Score Fusion

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Algorithm Design
**Status:** Accepted
**Date:** 2026-03-23

---

## Context

P5 compares three retriever types: BM25, dense, and hybrid. The hybrid retriever needs to combine scores from both. The problem is that their scales are completely different.

BM25 scores are unbounded. On a 500-chunk corpus of product PDFs, scores ranged from roughly 0 to 15 depending on term frequency and document length. Dense retrieval uses cosine similarity on L2-normalised vectors, which puts scores in [0, 1]. Averaging those two raw numbers directly does not make sense. A BM25 score of 8.3 would swamp a dense score of 0.87, meaning the hybrid behaves identically to BM25 for any query with keyword matches and only falls back to dense when nothing matches. That is not a hybrid retriever. It is a BM25 retriever with extra steps.

---

## Decision

Before combining, min-max normalise the BM25 scores to [0, 1]: `bm25_norm = (v - bm25_min) / (bm25_max - bm25_min)`. Combined score is `alpha * dense_score + (1 - alpha) * bm25_norm`, with `alpha=0.7` as the default. That default puts slightly more weight on semantic similarity, which performed better during manual spot-checks on the ShopTalk product FAQs. P5's hyperparameter sweep will test alpha at 0.3, 0.5, and 0.7.

One edge case: when no query keywords appear in any chunk, BM25 assigns identical scores to everything. Min-max with max == min is division by zero. The fix is to set all normalised scores to 0.0, which means no BM25 signal for that query and the combined score reduces to pure dense. That is the right fallback because BM25 has nothing useful to contribute.

---

## Alternatives Considered

**Reciprocal Rank Fusion (RRF)** - Rank-based, no normalisation needed, robust to outliers. But it loses score magnitude entirely. A dense result at 0.97 and one at 0.71 are both just rank 1 and rank 2. For an experiment platform where score distributions across configs matter, that information loss is a problem.

**Z-score normalisation** - Works better if you know the score distribution. Requires computing mean and standard deviation, which changes as chunks are added. Min-max over the candidate set is simpler and sufficient here.

**Raw combination without normalisation** - Fast to implement, obviously wrong. BM25 dominates by an order of magnitude on any query with keyword hits.

---

## Quantified Validation

- On manual spot-checks against 4 ShopTalk product PDFs, raw BM25 combination ranked identical to standalone BM25 on 23/25 queries. Min-max fusion gave meaningfully different rankings on 18/25, with reranker scores confirming better top-5 candidates.
- The identical-score edge case (`bm25_max == bm25_min`) occurs on approximately 8% of evaluation queries where no product keywords appear, confirming the guard matters.
- `HybridRetriever` is covered by 8 unit tests: one specifically exercises the all-identical BM25 score path to verify the 0.0 fallback.

---

## Consequences

`HybridRetriever` exposes `_alpha` so the experiment runner can sweep it as a hyperparameter. The `create_retriever` factory defaults to 0.7 but passes through whatever `ExperimentConfig.hybrid_alpha` specifies.

The min-max normalisation is per-query, over the candidate set returned by BM25, not a global normalisation across the whole corpus. Global normalisation would require a full corpus scan at query time, and scale differences between queries (short keyword queries vs. long natural language questions) make it less stable anyway.

If a different sparse retriever (e.g., SPLADE) is swapped in, the same normalisation approach applies as long as that retriever's scores are unbounded. `HybridRetriever` accepts any `BaseRetriever` for both sub-retrievers, so no changes needed at the composition layer.
