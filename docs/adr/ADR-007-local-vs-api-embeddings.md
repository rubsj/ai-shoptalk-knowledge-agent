# ADR-007: Local vs API Embeddings, Ollama nomic-embed-text

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Embedding Backend
**Status:** Accepted
**Date:** 2026-03-28

---

## Context

Days 1-4 tested three embedding backends: MiniLM (384d, local), mpnet (768d, local), and OpenAI `text-embedding-3-small` (1536d, API). OpenAI produced the best retrieval quality (best config NDCG@5=0.896) but at API cost (~$0.04/run) and with a network dependency. MiniLM and mpnet are free but require local model weights loaded into RAM.

The fourth backend to evaluate: **Ollama nomic-embed-text**, a 768d open-source model served locally via the Ollama REST API. It requires no Python model loading (Ollama manages the process), has zero API cost, keeps data local, and runs on the same hardware as mpnet. I wanted to know whether a locally-served REST-based embedding model could approach the quality of a cloud API at zero cost.

---

## Decision

**Add `ollama_nomic` as a supported embedding backend.** I ran 6 experiments (best 3 chunkers x dense + hybrid) and positioned it as the recommended option when API cost or data privacy is a constraint.

`src/embedders/ollama_embedder.py` implements `OllamaEmbedder` using `httpx` for REST calls to `http://localhost:11434/api/embeddings`. The class runs a health check at `__init__` time (`GET /api/tags`) and raises `OllamaUnavailableError` if the server isn't reachable. `OLLAMA_BASE_URL` env var overrides the default port for non-standard setups. The experiment runner catches `OllamaUnavailableError` and skips the embedder group with a log warning, so nothing crashes if Ollama isn't running.

---

## Alternatives Considered

**MiniLM (384d, local)** - Good quality (avg 0.732 NDCG@5), zero cost, full data privacy, ~500ms/query. No infrastructure dependency. Already a supported backend.

**mpnet (768d, local)** - Same profile as MiniLM at 768d, avg 0.734. Already supported.

**OpenAI text-embedding-3-small (1536d, API)** - Best quality (avg 0.840), but ~$0.04/run, data leaves the machine, requires network + API key. Already supported.

Ollama nomic fills a specific gap: same 768d as mpnet, zero cost like the local models, but served via REST rather than loaded into the Python process. It reuses the existing FAISS `IndexFlatIP` infrastructure without changes to the index pipeline.

---

## Quantified Validation

Results from 6 experiments (configs 47-52):

| Config | NDCG@5 | Recall@5 | MRR | Latency |
|--------|--------|----------|-----|---------|
| heading\_semantic + dense | 0.633 | 0.778 | 0.592 | ~2963ms |
| fixed + dense | 0.699 | 0.889 | 0.634 | ~1930ms |
| sliding\_window + dense | 0.636 | 0.778 | 0.620 | ~2052ms |
| heading\_semantic + hybrid | 0.723 | 0.833 | 0.696 | ~2353ms |
| fixed + hybrid | 0.735 | 0.889 | 0.699 | ~1670ms |
| **sliding\_window + hybrid** | **0.757** | **0.889** | **0.722** | **~1772ms** |

Versus equivalent mpnet configs (same 768d, same chunkers):

| Chunker | Retriever | mpnet | Ollama | Delta |
|---------|-----------|-------|--------|-------|
| heading\_semantic | Dense | 0.655 | 0.633 | -0.022 |
| fixed | Dense | 0.685 | 0.699 | +0.014 |
| sliding\_window | Dense | 0.619 | 0.636 | +0.017 |
| heading\_semantic | Hybrid | 0.768 | 0.723 | -0.046 |
| fixed | Hybrid | 0.710 | 0.735 | +0.026 |
| sliding\_window | Hybrid | 0.823 | 0.757 | -0.065 |

3 of 6 Ollama configs match or exceed the equivalent mpnet config. The widest gap is -0.065 NDCG@5 on sliding\_window + hybrid.

---

## Consequences

I get a zero-cost, fully local embedding option that works in air-gapped or restricted environments with no API key required. At scale (1000 runs), that's ~$40 saved versus OpenAI. The best Ollama config (0.757 NDCG@5) exceeds all MiniLM and mpnet dense configs, so it's not just a cost play.

The trade-off is infrastructure: Ollama needs to be running as a separate process, which MiniLM/mpnet don't require. The `OllamaUnavailableError` + graceful skip handles this cleanly, but it's still a dependency to manage. Sequential REST calls also add overhead: ~50ms per text via localhost HTTP vs batch `SentenceTransformer.encode()`. For 515 chunks that's ~26s vs ~14s for mpnet CPU batch. Fine for the experiment grid, but a production deployment would need batching.

The quality ceiling is real: best Ollama config (0.757) is -0.14 below best OpenAI config (0.896). For deployments where retrieval quality is the top priority, OpenAI is still the better backend.

Hybrid retrieval partially compensates for the quality gap: +0.090 to +0.121 NDCG@5 improvement over Ollama dense for heading\_semantic and sliding\_window chunkers.

The `device` parameter accepted by `OllamaEmbedder.__init__` is a no-op (Ollama manages its own device). I kept it to preserve the factory API contract without special-casing.
