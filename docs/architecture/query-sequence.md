# Query Pipeline

A user question flows through embedding, retrieval, optional reranking, LLM generation, and citation extraction. The full chain runs in under 3 seconds for the best config — most of that time is the LLM generation call, not retrieval.

The `opt` block for reranking is the most interesting architectural choice here. Reranking improved NDCG@5 by +0.1124 on average across all 8 configs tested, but added ~200ms latency per query. Every single config that added reranking improved — zero regressions. That made the trade-off obvious for quality-first deployments.

```mermaid
sequenceDiagram
    participant User
    participant CLI as serve.py
    participant Em as Embedder
    participant Ret as Retriever
    participant Rr as Reranker
    participant LLM as LiteLLMClient
    participant Cit as extract_citations()

    User->>CLI: question text

    rect rgb(240, 255, 240)
        Note over CLI,Em: Query Embedding (~50ms local, ~200ms API)
        CLI->>Em: embed_query(question)
        Em-->>CLI: query_vector (D,)
    end

    rect rgb(240, 248, 255)
        Note over CLI,Ret: Retrieval (~10ms dense, ~5ms BM25)
        CLI->>Ret: retrieve(question, top_k=5)
        Note right of Ret: Dense: FAISS inner product search<br/>BM25: keyword matching on tokenized corpus<br/>Hybrid: α·dense + (1-α)·bm25_norm
        Ret-->>CLI: list[RetrievalResult]
    end

    opt use_reranking = true
        rect rgb(255, 248, 240)
            Note over CLI,Rr: Reranking (~200ms CrossEncoder, ~100ms Cohere API)
            CLI->>Rr: rerank(question, results, top_k=5)
            Note right of Rr: Scores each (query, chunk) pair independently<br/>O(k) forward passes — expensive but accurate
            Rr-->>CLI: reranked list[RetrievalResult]
        end
    end

    rect rgb(245, 245, 220)
        Note over CLI,LLM: Answer Generation (~1500ms via GPT-4o-mini)
        CLI->>CLI: build_qa_prompt(question, context_chunks)
        Note right of CLI: Numbered context: [1] chunk1 [2] chunk2 ...<br/>Prompt instructs LLM to cite with [N] markers
        CLI->>LLM: generate(prompt)
        LLM-->>CLI: answer text with [N] citations
    end

    rect rgb(255, 240, 245)
        Note over CLI,Cit: Citation Extraction (<1ms)
        CLI->>Cit: extract_citations(answer, context_chunks)
        Note right of Cit: Regex parses [N] markers<br/>Validates N is within chunk range<br/>Deduplicates, maps to Citation objects
        Cit-->>CLI: list[Citation]
    end

    CLI-->>User: Answer + source chunks with page numbers
```

## Data Flow

| Stage | Input | Output | Key Type |
|-------|-------|--------|----------|
| Embed query | `str` | `np.ndarray` | Shape: (D,) — L2-normalized, same space as indexed chunks |
| Retrieve | query + top_k | `list[RetrievalResult]` | `RetrievalResult(chunk, score, retriever_type, rank)` |
| Rerank | query + results + top_k | `list[RetrievalResult]` | Re-scored by cross-encoder, reordered by new scores |
| Generate | prompt with numbered context | `str` | Answer with `[N]` citation markers referencing context chunks |
| Extract citations | answer + chunks | `list[Citation]` | `Citation(chunk_id, source, page_number, text_snippet)` |

## Latency Budget (best config, heading_semantic + openai + dense)

| Stage | Time | % of Total |
|-------|------|-----------|
| Query embedding | ~200ms | ~7% |
| FAISS retrieval | ~10ms | <1% |
| Reranking (if enabled) | ~200ms | ~7% |
| LLM generation | ~1500ms | ~85% |
| Citation extraction | <1ms | <1% |
| **Total (no reranking)** | **~1700ms** | |
| **Total (with reranking)** | **~1900ms** | |
