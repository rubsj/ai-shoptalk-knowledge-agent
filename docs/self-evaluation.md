# Self-Evaluation

Answers to the 6 questions from PRD Section 8c, backed by experiment data from `results/comparison_report.md` and `results/experiments/`.

---

## Q1: Can you explain why configuration X outperformed configuration Y?

The best configuration `heading_semantic_openai_dense` (NDCG@5 = 0.896, experiment `d702830f`) outperformed the worst `embedding_semantic_mpnet_hybrid` (NDCG@5 = 0.607, experiment `b6596d30`) by +0.289 NDCG@5. Three factors explain the gap, in order of impact:

**Embedding model dominates.** Switching from mpnet to OpenAI is the single largest lever: the iteration log shows embedding_model changes produce NDCG@5 deltas of +0.14 to +0.24. OpenAI's 1536-dimensional embeddings capture more semantic nuance in academic language than mpnet's 768d. This held across every chunker/retriever combination — not a single config where mpnet beat OpenAI on NDCG@5.

**Chunking strategy matters second.** `heading_semantic` (NDCG@5 = 0.7752 avg) outperformed `embedding_semantic` (0.6874 avg) by +0.088. Heading-aware splitting preserves section boundaries that academic papers use to organize arguments. The embedding-semantic chunker relies on MiniLM cosine similarity to detect boundaries, which misses structural heading cues.

**Retriever type is secondary.** Dense retrieval slightly edges out hybrid for the best config (0.896 vs 0.870, per experiment `2964c02f`). This was unexpected — hybrid wins on average across all configs (0.7515 vs 0.7176), but the OpenAI embeddings are good enough that BM25's keyword signal doesn't add value at the top of the ranking.

Evidence chain: iteration log entries #1-2 (embedder impact), Q1 table in comparison report (chunker averages), traceability table (dense vs hybrid for best config).

---

## Q2: Do your metrics align with qualitative assessment of answers?

Yes. The LLM judge (5-axis, GPT-4o-mini) scored the best config at 4.77/5.0 overall, which aligns with the high retrieval metrics (NDCG@5 = 0.896, Recall@5 = 1.0). The per-axis breakdown:

| Axis | Score |
|------|-------|
| Relevance | 4.83 |
| Accuracy | 4.78 |
| Completeness | 4.72 |
| Conciseness | 4.78 |
| Citation Quality | 4.72 |

Citation Quality (4.72) correlates with Recall@5 (1.0): when all relevant chunks are retrieved, the LLM has better source material to cite from. Completeness (4.72) is the lowest axis, which makes sense — even with perfect retrieval, the LLM occasionally summarizes rather than synthesizing across all retrieved chunks.

The alignment isn't circular. Retrieval metrics measure whether the right chunks were found. Judge metrics measure whether the generated answer is useful. They could diverge — a config could retrieve perfectly but generate poorly (hallucination), or retrieve mediocrely but generate well (LLM compensates). That they agree here validates both the retrieval pipeline and the generation prompt design.

---

## Q3: Does reranking improve top-3 results even if top-5 metrics are similar?

I only measured @5 metrics — the experiment grid ran with `top_k=5` throughout, and I don't have dedicated @3 evaluation data. Running @3 would require re-executing the grid with `top_k=3`, which I didn't do.

What I can say from the @5 data: reranking improved MRR by +0.1473 on average across all 8 reranking configs. MRR measures the reciprocal rank of the *first* relevant result, so this improvement directly reflects top-of-list quality. If the first relevant chunk moves from position 3 to position 1, that matters for top-3 display even though Recall@5 might not change.

The per-config reranking deltas (NDCG@5):

| Config | Base | Reranked | Delta |
|--------|------|----------|-------|
| recursive_minilm_dense + cross_encoder | 0.642 | 0.762 | +0.121 |
| recursive_minilm_hybrid + cross_encoder | 0.693 | 0.820 | +0.127 |
| recursive_minilm_dense + cohere | 0.642 | 0.766 | +0.125 |
| recursive_minilm_hybrid + cohere | 0.693 | 0.811 | +0.118 |
| recursive_mpnet_dense + cross_encoder | 0.639 | 0.730 | +0.091 |
| recursive_mpnet_hybrid + cross_encoder | 0.679 | 0.802 | +0.122 |
| recursive_mpnet_dense + cohere | 0.639 | 0.721 | +0.083 |
| recursive_mpnet_hybrid + cohere | 0.679 | 0.792 | +0.113 |

All 8 improved — zero regressions. NDCG weighs higher-ranked results more heavily (logarithmic discount), so the +0.1124 avg NDCG@5 improvement confirms that reranking is pushing relevant chunks toward the top positions, not just shuffling the tail.

---

## Q4: Are citations accurate and helpful for verification?

Citations use index-reference format: the generation prompt numbers each context chunk as `[1]`, `[2]`, etc., and instructs the LLM to cite with `[N]` markers. The `extract_citations()` function in `src/generator.py:84` parses these markers with regex, validates that N is within the chunk range, deduplicates, and maps each to a `Citation` object with `chunk_id`, `source` (PDF filename), `page_number`, and `text_snippet`.

Accuracy has two layers:

1. **Structural accuracy** (parse-only, validated in the experiment grid): Does the citation reference a real chunk? The parser rejects out-of-range markers with a warning. This works reliably — GPT-4o-mini produces valid `[N]` markers when instructed.

2. **Semantic accuracy** (validated by LLM judge): Does the cited chunk actually support the claim? The Citation Quality axis scored 4.72/5.0, indicating the LLM almost always cites the right source. The occasional miss is citing a contextually adjacent chunk rather than the most directly relevant one.

For verification, each citation links back to a specific chunk with page number and text snippet. A user can click through to the source PDF page. This is more useful than vague "based on the provided documents" responses.

---

## Q5: Can the system handle edge cases?

Tested across 4 academic papers with varying characteristics:

**Document length.** Papers range from ~15K chars (Sentence-BERT) to ~39K chars (RAG survey). All 5 chunking strategies handle both extremes. Heading-semantic produces 8 chunks for short papers and 40+ for long ones. Fixed-size chunking is length-agnostic by design.

**Ambiguous queries.** The ground truth includes queries that require synthesis across multiple paper sections (e.g., "How does the attention mechanism relate to the transformer architecture?"). The system retrieves chunks from different sections, and the LLM synthesizes. Judge scores on these queries average 4.6 — slightly lower than factoid queries (4.8) but still well above the 4.0 target.

**No keyword overlap.** BM25 returns empty results when query terms don't appear in any chunk. The hybrid retriever handles this gracefully — if all BM25 scores are identical (zero keyword matches), normalized scores become 0.0 and the hybrid falls back to pure dense retrieval. This edge case is handled in `src/retrievers/hybrid.py` and tested in `tests/test_retrievers.py`.

**Ollama unavailable.** If the Ollama service isn't running, `OllamaEmbedder.__init__` raises `OllamaUnavailableError` with a clear message. The experiment runner catches this and skips Ollama configs, continuing the grid with other embedders.

**Integration tests** in `tests/test_integration.py` verify the full pipeline end-to-end: PDF extraction through answer generation, confirming that no component silently drops data.

---

## Q6: How do local embeddings compare to API embeddings, and what would you recommend for production?

Six Ollama configs tested: best 3 chunkers (heading_semantic, fixed, sliding_window) x 2 retrievers (dense, hybrid).

**Quality gap is real but manageable.** Best Ollama config (`sliding_window_ollama_nomic_hybrid`) achieves NDCG@5 = 0.757. Best OpenAI config achieves 0.896. That's a -0.139 gap. But Ollama with hybrid retrieval meets the PRD target of NDCG@5 > 0.75.

**Ollama competes with mpnet, not OpenAI.** Both are 768-dimensional. In head-to-head comparisons on the same chunker+retriever combos, Ollama matches or exceeds mpnet in 3 of 6 configs. The widest gap is -0.065 (sliding_window + hybrid). Hybrid retrieval consistently boosts Ollama: +0.090 to +0.121 NDCG@5 across all chunkers.

**Cost and latency.** Ollama: $0 embedding cost, ~1700-2900ms latency (sequential REST calls to localhost). OpenAI: ~$0.008/run, ~476ms latency (batched API call). Ollama is slower per-query but free at any scale. At 1000 experiment runs, that's $0 vs ~$8 in embedding costs alone.

**Production recommendation:**

- **Cost-sensitive or privacy-constrained deployments:** Ollama nomic-embed-text with hybrid retrieval. Quality is acceptable (NDCG@5 >= 0.75), data never leaves the machine, and embedding cost is zero. Pair with hybrid retrieval to compensate for the quality gap.

- **Quality-first deployments:** OpenAI text-embedding-3-small with dense retrieval. The -0.139 NDCG@5 gap is significant for applications where answer quality directly affects user trust (e.g., medical, legal). The $0.008/run cost is negligible for most production workloads.

- **Hybrid approach for production:** Index with both. Use Ollama for bulk/batch operations (overnight re-indexing, dev/test), OpenAI for user-facing real-time queries. The factory pattern makes switching a config change, not a code change.
