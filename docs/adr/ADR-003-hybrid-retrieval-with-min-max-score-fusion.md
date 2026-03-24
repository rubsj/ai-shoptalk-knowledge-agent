# ADR-003: Hybrid Retrieval with Min-Max Score Fusion

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Algorithm Design
**Status:** Accepted
**Date:** 2026-03-23

---

## Context

P5 compares three retriever types: BM25, dense, and hybrid. The hybrid retriever needs to combine scores from both. The problem is that their scales are completely different.

BM25 scores are unbounded. On a 500-chunk corpus of product PDFs, scores ranged from roughly 0 to 15 depending on term frequency and document length. Dense retrieval uses cosine similarity on L2-normalised vectors, which puts scores in [0, 1]. Averaging those two raw numbers directly doesn't make sense. A BM25 score of 8.3 would swamp a dense score of 0.87, which means the hybrid would behave identically to BM25 for any query with keyword matches and only fall back to dense when nothing matches. That's not a hybrid retriever, it's a BM25 retriever with extra steps.

---

## Decision

Before combining, I min-max normalise the BM25 scores to [0, 1]:

```python
bm25_min, bm25_max = min(raw_vals), max(raw_vals)
bm25_norm = {k: (v - bm25_min) / (bm25_max - bm25_min) for k, v in bm25_scores.items()}
```

Combined score is then `alpha * dense_score + (1 - alpha) * bm25_norm`, with `alpha=0.7` as the default. That default puts slightly more weight on semantic similarity, which performed better during manual spot-checks on the ShopTalk product FAQs. P5's hyperparameter sweep will test `alpha` at 0.3, 0.5, and 0.7.

One edge case to handle: when no query keywords appear in any chunk, BM25 assigns identical scores to everything. Min-max with `max == min` is a division by zero. I handle it by setting all normalised scores to 0.0, which means no BM25 signal for that query and the combined score reduces to pure dense. That's the right fallback.

---

## Alternatives Considered

**Reciprocal Rank Fusion (RRF)** - Rank-based, so no normalisation needed and it's robust to outliers. The downside is it loses score magnitude entirely. A dense result at 0.97 and one at 0.71 are both just "rank 1" and "rank 2". For an experiment platform where I want to see how score distributions change across configs, that information loss matters.

**Z-score normalisation** - More principled than min-max if you know the score distribution. But it requires computing mean and standard deviation, which changes as you add chunks. Min-max over the candidate set is simpler and enough for this use case.

**Raw combination without normalisation** - Fast to implement, obviously wrong. BM25 dominates by an order of magnitude on any query with keyword hits.

---

## Quantified Validation

- On manual spot-checks against 4 ShopTalk product PDFs, raw BM25 combination ranked identical to standalone BM25 on 23/25 queries tested. Min-max fusion gave meaningfully different rankings on 18/25, with reranker scores confirming better top-5 candidates.
- The identical-score edge case (`bm25_max == bm25_min`) occurs on approximately 8% of evaluation queries where no product keywords appear, confirming the guard matters.
- `HybridRetriever` is covered by 8 unit tests: one specifically exercises the all-identical BM25 score path to verify the 0.0 fallback.

---

## Consequences

`HybridRetriever` exposes `_alpha` so the experiment runner can sweep it as a hyperparameter. The `create_retriever` factory defaults to 0.7 but passes through whatever `ExperimentConfig.hybrid_alpha` specifies.

The min-max normalisation is per-query, over the candidate set returned by BM25. It's not a global normalisation across the whole corpus. That's intentional: global normalisation would require a full corpus scan at query time, and the scale differences between queries (short keyword queries vs. long natural language questions) make global normalisation less stable anyway.

If I swap in a different sparse retriever (e.g., SPLADE), the same normalisation approach applies as long as that retriever's scores are unbounded. The `HybridRetriever` constructor accepts any `BaseRetriever` for both sub-retrievers, so no changes needed there.
