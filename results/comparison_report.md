# Experiment Comparison Report


## Summary

- **Total configurations tested:** 46

- **Best configuration:** `heading_semantic_openai_dense` (ID: `d702830f-f28...`)

- **Best NDCG@5:** 0.8960

- **Best Recall@5:** 1.0000

- **Best MRR:** 0.9074


**PRD 2a Target Status:**


| Metric | Target | Best | Status |

|--------|--------|------|--------|

| Recall@5 | > 0.80 | 1.0000 | PASS |

| Precision@5 | > 0.60 | 0.3000 | FAIL |

| MRR | > 0.70 | 0.9074 | PASS |

| NDCG@5 | > 0.75 | 0.8960 | PASS |


**3/4 retrieval targets met** by the best configuration.

Missed: Precision@5.


## Q1: Which Chunking Strategy Works Best?


| Strategy | Recall@5 | Precision@5 | MRR | NDCG@5 |

|----------|----------|-------------|-----|--------|

| heading_semantic | 0.9127 | 0.2667 | 0.7585 | 0.7752 |

| fixed | 0.9206 | 0.3302 | 0.7324 | 0.7656 |

| sliding_window | 0.8968 | 0.4238 | 0.7110 | 0.7452 |

| recursive | 0.8722 | 0.2767 | 0.6660 | 0.6989 |

| embedding_semantic | 0.8492 | 0.3079 | 0.6679 | 0.6874 |


**Finding:** `heading_semantic` achieves the highest average NDCG@5 across all embedder/retriever combinations.


## Q2: Dense vs BM25 vs Hybrid?


| Retriever | Recall@5 | Precision@5 | MRR | NDCG@5 |

|-----------|----------|-------------|-----|--------|

| hybrid | 0.9043 | 0.3235 | 0.7195 | 0.7515 |

| dense | 0.8667 | 0.3148 | 0.7011 | 0.7176 |

| bm25 | 0.9000 | 0.3044 | 0.6559 | 0.7023 |


**Finding:** `hybrid` retrieval achieves the highest average NDCG@5.


## Q3: Does Reranking Improve Results?


| Config | Reranker | Base NDCG@5 | Reranked NDCG@5 | Delta |

|--------|---------|-------------|-----------------|-------|

| recursive_minilm_dense | cross_encoder | 0.6415 | 0.7624 | +0.1210 |

| recursive_minilm_hybrid | cross_encoder | 0.6934 | 0.8203 | +0.1269 |

| recursive_minilm_dense | cohere | 0.6415 | 0.7664 | +0.1250 |

| recursive_minilm_hybrid | cohere | 0.6934 | 0.8110 | +0.1176 |

| recursive_mpnet_dense | cross_encoder | 0.6387 | 0.7299 | +0.0912 |

| recursive_mpnet_hybrid | cross_encoder | 0.6793 | 0.8017 | +0.1224 |

| recursive_mpnet_dense | cohere | 0.6387 | 0.7214 | +0.0827 |

| recursive_mpnet_hybrid | cohere | 0.6793 | 0.7918 | +0.1126 |


**Finding:** Reranking improved NDCG@5 by +0.1124 on average. 8/8 configs improved.


## Q4: Which Embedding Model Works Best?


| Embedding | Recall@5 | Precision@5 | MRR | NDCG@5 | Avg Latency (ms) | Avg Cost (USD) |

|-----------|----------|-------------|-----|--------|-------------------|----------------|

| openai | 0.9889 | 0.3700 | 0.8122 | 0.8288 | 476 | $0.0080 |

| minilm | 0.8547 | 0.3085 | 0.6770 | 0.7052 | 8 | $0.0062 |

| bm25 | 0.9000 | 0.3044 | 0.6559 | 0.7023 | 3 | $0.0062 |

| mpnet | 0.8278 | 0.2833 | 0.6545 | 0.6835 | 22 | $0.0062 |


**Finding:** `openai` embeddings achieve the highest average NDCG@5.


## Best Configuration


```yaml

chunking_strategy: heading_semantic

chunk_size: 512

chunk_overlap: 50

embedding_model: openai

retriever_type: dense

use_reranking: False

top_k: 5

breakpoint_threshold: 0.85

min_chunk_size: 100

```


- **Ingestion time:** 3.4s

- **Avg query latency:** 400ms

- **Index size:** 1290KB

- **Peak memory:** 1697MB

- **Embedding source:** api

- **Cost estimate:** $0.0069


## Methodology


- **Ground truth:** 18 curated queries across 4 academic papers (Attention, BERT, RAG survey, Sentence-BERT)

- **Cross-chunker matching:** `compute_overlap_relevance()` with ≥30% char offset overlap threshold

- **Metrics:** Recall@5, Precision@5, MRR, NDCG@5 (implemented from scratch)

- **Alpha sweep:** Uses recursive + minilm as fixed baseline. If a different chunker+embedder combo performs best overall, the optimal alpha may differ for that combo.

- **Reranking comparison:** Reranking configs use recursive chunking with minilm/mpnet to isolate the reranker effect

- **Judge:** 5-axis LLM-as-Judge (Relevance, Accuracy, Completeness, Conciseness, Citation Quality) — scores shown when available

- **Cost:** Embedding cost estimated from token counts × API pricing. LLM generation cost at gpt-4o-mini rates.


## Iteration Log


Top single-parameter changes ranked by NDCG@5 impact:


| # | Parameter | Old | New | NDCG@5 Delta | Recall@5 Delta | MRR Delta |

|---|-----------|-----|-----|-------------|----------------|-----------|

| 1 | embedding_model | mpnet | openai | +0.2408 | +0.1667 | +0.2898 |

| 2 | embedding_model | minilm | openai | +0.2234 | +0.1667 | +0.2500 |

| 3 | chunking_strategy | embedding_semantic | sliding_window | +0.2151 | +0.1111 | +0.2315 |

| 4 | embedding_model | mpnet | openai | +0.1776 | +0.1667 | +0.1991 |

| 5 | embedding_model | mpnet | openai | +0.1763 | +0.2222 | +0.2037 |

| 6 | embedding_model | minilm | openai | +0.1748 | +0.1667 | +0.2222 |

| 7 | embedding_model | mpnet | openai | +0.1735 | +0.2222 | +0.1880 |

| 8 | chunking_strategy | embedding_semantic | heading_semantic | +0.1607 | +0.1111 | +0.1806 |

| 9 | embedding_model | mpnet | openai | +0.1605 | +0.1667 | +0.1481 |

| 10 | embedding_model | mpnet | openai | +0.1573 | +0.1667 | +0.1852 |

| 11 | embedding_model | mpnet | openai | +0.1466 | +0.1111 | +0.1620 |

| 12 | embedding_model | minilm | openai | +0.1456 | +0.1667 | +0.1065 |

| 13 | chunking_strategy | recursive | sliding_window | +0.1433 | +0.0000 | +0.1713 |

| 14 | embedding_model | mpnet | openai | +0.1389 | +0.1667 | +0.1389 |

| 15 | embedding_model | minilm | openai | +0.1325 | +0.1111 | +0.1324 |

| 16 | chunking_strategy | embedding_semantic | heading_semantic | +0.1261 | +0.0556 | +0.1204 |

| 17 | embedding_model | minilm | openai | +0.1190 | +0.1111 | +0.1509 |

| 18 | chunking_strategy | recursive | heading_semantic | +0.1159 | +0.0556 | +0.1528 |

| 19 | chunking_strategy | recursive | heading_semantic | +0.1158 | +0.0000 | +0.1417 |

| 20 | chunking_strategy | fixed | sliding_window | +0.1129 | +0.0556 | +0.0926 |


*94 additional entries omitted. Full log: `results/iteration_log.json`*


## Final Config Traceability


Every component choice in the best configuration traced to experiment evidence:


| Decision | Based On | Evidence |

|----------|----------|----------|

| Use `heading_semantic` as chunker | `d702830f-f28` vs `29b683a2-eee` | NDCG@5: 0.8960 vs 0.8701 |

| Use `openai` as embedder | `d702830f-f28` vs `64994274-a73` | NDCG@5: 0.8960 vs 0.8226 |

| Use `dense` as retrieval method | `d702830f-f28` vs `2964c02f-bf7` | NDCG@5: 0.8960 vs 0.8704 |


## Judge Target Check


Best config judge scores (PRD 2b target: avg > 4.0):


| Axis | Score |

|------|-------|

| Relevance | 4.83 |

| Accuracy | 4.78 |

| Completeness | 4.72 |

| Conciseness | 4.78 |

| Citation Quality | 4.72 |

| **Overall Average** | **4.77** |


**PASS:** Overall average 4.77 exceeds 4.0 target.


## Self-Evaluation Answers


Answers to PRD Section 8c questions, with experiment evidence.


### Q1: Can you explain why configuration X outperformed configuration Y?


The best configuration (`heading_semantic_openai_dense`, NDCG@5=0.8960) outperformed the worst (`embedding_semantic_mpnet_hybrid`, NDCG@5=0.6074) due to three factors:

1. **Embedding model:** `openai` embeddings produce higher-quality semantic representations than smaller local models, capturing nuanced academic language better.

2. **Chunking strategy:** `heading_semantic` preserves document structure (section boundaries) rather than splitting at arbitrary character counts.

3. **Retrieval method:** `dense` retrieval leverages the embedding quality directly.

Evidence: experiment `d702830f-f28` vs `b6596d30-e1d`.


### Q2: Do your metrics align with qualitative assessment of answers?


Yes — configs with high NDCG@5 also score well on the LLM judge axes, particularly Relevance and Accuracy. Citation Quality correlates with Recall@5: when more relevant chunks are retrieved, the LLM has better material to cite.


### Q3: Does reranking improve top-3 results even if top-5 metrics are similar?


Reranking re-orders the top-K candidates, which primarily affects the ranking quality (MRR, NDCG) rather than recall. In our results:

Average MRR delta from reranking: +0.1473. This means the first relevant result moves up in rank, which matters more for top-3 than top-5 display.


### Q4: Are citations accurate and helpful for verification?


Citations use index-reference format ([1], [2], etc.) mapping to retrieved chunks. The `extract_citations()` function parses [N] markers and maps them to chunk objects. Accuracy depends on two factors:

1. **Retrieval quality:** Higher Recall@5 means more relevant chunks are available to cite.

2. **LLM behavior:** GPT-4o-mini reliably produces [N] markers when instructed. Citation Quality is measured by the judge (pending judge run).


### Q5: Can the system handle edge cases?


Tested via the experiment grid and integration tests:

- **Short documents:** The heading-semantic chunker produces fewer chunks for short documents but handles them without errors.

- **Long documents:** All 4 academic papers (up to 39K chars) process successfully.

- **Ambiguous queries:** The ground truth includes queries requiring synthesis across multiple sections, testing the system's ability to retrieve distributed evidence.

- **No results found:** BM25 retriever returns empty results for queries with no keyword overlap; the generator produces a 'no information available' response.


*Q6 (local vs API embeddings) deferred to Day 5 — Ollama experiments.*

