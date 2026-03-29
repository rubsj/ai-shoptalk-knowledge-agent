# P5 Day 5 Execution Plan — Local Model Experiments + Concept Deep-Dive

## Context

Days 0-4 complete: 46 configs tested, 562 tests, 94-97% coverage, 10 charts, Q1-Q4 answered, ADR-001 through ADR-006 written. Best config: `heading_semantic_openai_dense` (NDCG@5=0.896, Recall@5=1.0, MRR=0.907).

Day 5 adds Ollama (nomic-embed-text, 768d) as the 4th embedding backend, runs 6 new experiments (best 3 chunkers × dense + hybrid), produces Chart 11, answers Q5, writes ADR-007, concept library entries, and learning journal.

**Repo:** `/Users/rubyjha/repo/AI/ai-shoptalk-knowledge-agent`
**Branch:** `feat/p5-day5-ollama` from `main`

---

## Phase 1: OllamaEmbedder + Configs + Tests

### Step 1.1 — Add `"ollama_nomic"` to schema types

**File:** `src/schemas.py:29`
```python
# Before:
EmbeddingModel = Literal["minilm", "mpnet", "openai"]
# After:
EmbeddingModel = Literal["minilm", "mpnet", "openai", "ollama_nomic"]
```

### Step 1.2 — Create OllamaEmbedder class

**New file:** `src/embedders/ollama_embedder.py`

Follow the pattern from `src/embedders/minilm.py`: class constants, eager/health-check init, `embed()` with L2 normalization via `faiss.normalize_L2()`, `embed_query()` reuses `embed()`, `dimensions` property.

```python
"""Ollama nomic-embed-text embedder — 768d local embeddings via REST API.

Zero cost, no network dependency (runs locally via Ollama).
API: POST http://localhost:11434/api/embeddings {"model": "nomic-embed-text", "prompt": text}
Health check: GET http://localhost:11434/api/tags (fail fast if Ollama is down).
"""

import faiss
import httpx
import numpy as np
from src.interfaces import BaseEmbedder

class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama is not reachable. Callers catch and skip."""
    pass

class OllamaEmbedder(BaseEmbedder):
    _MODEL_NAME = "nomic-embed-text"
    _DIMENSIONS = 768
    _BASE_URL = "http://localhost:11434"

    def __init__(self, device: str | None = None) -> None:
        # device param accepted for factory API compatibility, ignored (Ollama manages its own device)
        import os
        self._base_url = os.getenv("OLLAMA_BASE_URL", self._BASE_URL)
        # Health check — fail fast
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            raise OllamaUnavailableError(f"Ollama not available at {self._base_url}: {e}") from e

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._DIMENSIONS), dtype=np.float32)
        vectors = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._MODEL_NAME, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        embeddings = np.array(vectors, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimensions(self) -> int:
        return self._DIMENSIONS
```

### Step 1.3 — Register in `__init__.py` and factory

**File:** `src/embedders/__init__.py` — Add import + export:
```python
from src.embedders.ollama_embedder import OllamaEmbedder
__all__ = ["MiniLMEmbedder", "MpnetEmbedder", "OpenAIEmbedder", "OllamaEmbedder"]
```

**File:** `src/factories.py:58-68` — Add branch in `create_embedder()`:
```python
def create_embedder(model_name: str, device: str | None = None) -> BaseEmbedder:
    if model_name == "minilm":
        return MiniLMEmbedder(device=device)
    if model_name == "mpnet":
        return MpnetEmbedder(device=device)
    if model_name == "openai":
        from src.embedders.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    if model_name == "ollama_nomic":                          # NEW
        from src.embedders.ollama_embedder import OllamaEmbedder
        return OllamaEmbedder(device=device)
    raise ValueError(f"Unknown embedding model: {model_name!r}")
```

**File:** `src/factories.py:21` — Add OllamaEmbedder to import (or lazy import in factory as shown above).

### Step 1.4 — Update experiment runner embedder order

**File:** `src/experiment_runner.py:48`
```python
# Before:
_EMBEDDER_ORDER: list[str | None] = ["minilm", "mpnet", None, "openai"]
# After:
_EMBEDDER_ORDER: list[str | None] = ["minilm", "mpnet", None, "ollama_nomic", "openai"]
```

Also add cost constant for Ollama (= $0):
```python
_OLLAMA_EMBED_COST_PER_TOKEN = 0.0  # local, zero cost
```

### Step 1.5 — Add `httpx` dependency

**File:** `pyproject.toml` — Add `"httpx>=0.27.0"` to dependencies. Verified: httpx is NOT currently installed (not a transitive dep despite LiteLLM being present).

### Step 1.6 — Create 6 YAML experiment configs

**Directory:** `experiments/configs/` — Add 6 new files (numbered 47-52, following existing convention).

Best 3 chunkers from Day 4 (verified from actual experiment data):
1. heading_semantic: avg NDCG@5 = 0.7643 (n=18)
2. fixed: avg NDCG@5 = 0.7521 (n=17)
3. sliding_window: avg NDCG@5 = 0.7406 (n=18)

| File | chunking_strategy | embedding_model | retriever_type | hybrid_alpha |
|------|-------------------|-----------------|----------------|-------------|
| `47_heading_ollama_dense.yaml` | heading_semantic | ollama_nomic | dense | — |
| `48_fixed_ollama_dense.yaml` | fixed | ollama_nomic | dense | — |
| `49_sliding_ollama_dense.yaml` | sliding_window | ollama_nomic | dense | — |
| `50_heading_ollama_hybrid.yaml` | heading_semantic | ollama_nomic | hybrid | 0.7 |
| `51_fixed_ollama_hybrid.yaml` | fixed | ollama_nomic | hybrid | 0.7 |
| `52_sliding_ollama_hybrid.yaml` | sliding_window | ollama_nomic | hybrid | 0.7 |

All set: `chunk_size: 512`, `chunk_overlap: 50`, `top_k: 5`, `use_reranking: false` (Cohere rate limit 10/min — skip reranking for Ollama configs).

Example (`47_heading_ollama_dense.yaml`):
```yaml
chunking_strategy: heading_semantic
chunk_size: 512
chunk_overlap: 50
top_k: 5
use_reranking: false
embedding_model: ollama_nomic
retriever_type: dense
```

### Step 1.7 — Tests

**New file:** `tests/test_ollama_embedder.py`
- `test_health_check_success` — mock httpx GET → 200, init succeeds
- `test_health_check_failure` — mock ConnectError, assert `OllamaUnavailableError`
- `test_embed_shape` — mock POST → 768d vectors, verify `(n, 768)` shape
- `test_embed_l2_normalized` — verify all row norms ≈ 1.0
- `test_embed_empty` — returns `(0, 768)`
- `test_embed_query_shape` — returns `(768,)`
- `test_factory_creates_ollama` — `create_embedder("ollama_nomic")` returns `OllamaEmbedder`

**Update existing tests:**
- `tests/test_factories.py` — if parametrized over embedding models, add mock/skip for `ollama_nomic`
- `tests/test_configs.py` — verify new YAML configs load without validation errors

### Step 1.7 — Validation
```bash
uv run pytest tests/test_ollama_embedder.py -v           # new tests green
uv run pytest tests/ -v --tb=short                       # all 562+ tests green
uv run pytest tests/ --cov=src --cov-report=term-missing # ≥94% coverage
```

---

## Phase 2: Run Experiments + Chart 11 + Q5

### Step 2.1 — Run 6 Ollama experiments

Use the existing `experiment_runner.py` orchestrator. The runner already:
- Groups configs by embedder (new `"ollama_nomic"` group will be picked up automatically)
- Builds FAISS indices per config
- Creates BM25 retriever for hybrid configs
- Computes R@K/P@K/MRR/NDCG@5 via `compute_overlap_relevance()`
- Runs LLM judge (5-axis scoring)
- Records performance metrics (latency, memory, cost, index size)

**Execution:**
```bash
uv run python -m src.experiment_runner \
  --config-dir experiments/configs/ \
  --output-dir results/experiments/ \
  --ground-truth data/ground_truth.json
```

Or run only the 6 new configs by filtering:
```bash
# If runner supports --filter or just run configs 47-52 in a subdirectory
```

**Graceful skip:** Wrap `create_embedder("ollama_nomic")` in the runner's embedder-group loop with try/except `OllamaUnavailableError` → log warning, skip group.

**Output:** 6 new JSON files in `results/experiments/` + updated `summary.json`.

### Step 2.2 — Update iteration_log.json

Append Ollama experiment entries to `results/iteration_log.json` (currently 114 entries). Each entry compares Ollama results against the equivalent non-Ollama config (e.g., heading_ollama_dense vs heading_mpnet_dense — same chunker, same retriever, different 768d embedder).

### Step 2.3 — Chart 11: Local vs API embedding comparison

**File:** `src/visualization.py` — Add new function after existing chart functions:

```python
def plot_local_vs_api_comparison(
    results: list[ExperimentResult],
    output_dir: Path,
) -> Path:
    """Chart 11: 3-panel figure — quality + latency + cost across 4 embedding models.

    Panel 1 (Quality): Grouped bars — NDCG@5 by model (MiniLM, mpnet, Ollama, OpenAI)
        for heading_semantic chunker, dense retrieval (controlled comparison)
    Panel 2 (Latency): Horizontal bars — avg query latency (ms) per model
    Panel 3 (Cost): Bar chart — estimated cost per run ($)

    Data: filter results for heading_semantic + dense configs across 4 embedders.
    """
```

Save to `results/charts/local_vs_api_comparison.png`.

Update `generate_all_charts()` to include Chart 11.

### Step 2.4 — Q5 in comparison report

**File:** `results/comparison_report.md` — Append Q5 section:

```markdown
## Q5: Do Local Embeddings Match API Quality?

### Setup
- Tested nomic-embed-text (768d) via Ollama against MiniLM (384d), mpnet (768d), OpenAI (1536d)
- 6 experiments: best 3 chunkers × {dense, hybrid}
- Same ground truth (18 queries), same judge, same metrics

### Results
| Config | Model | Retriever | NDCG@5 | Recall@5 | MRR | Latency | Cost |
[actual data from experiments]

### Hybrid Search Impact
[Dense vs hybrid delta for Ollama configs]

### Verdict
[Data-driven: quality gap quantified, cost/latency trade-off, when to use local vs API]
```

### Step 2.5 — Validation
```bash
ls results/experiments/ | grep ollama | wc -l   # → 6 files
ls results/charts/local_vs_api_comparison.png    # exists
grep "Q5" results/comparison_report.md           # found
```

---

## Phase 3: ADR-007 + Concept Library + Learning Journal

### Step 3.1 — ADR-007: Local vs API Embeddings

**New file:** `docs/adr/ADR-007-local-vs-api-embeddings.md`

Follow ADR-006 format:
```markdown
# ADR-007: Local vs API Embeddings — Ollama nomic-embed-text Results

**Date**: 2026-03-28
**Status**: Accepted
**Project**: P5 — ShopTalk Knowledge Agent

## Context
[Why test local embeddings: zero cost, no API dependency, data privacy, offline capability.
Ollama serves nomic-embed-text (768d, same dims as mpnet) via localhost REST API.]

## Decision
[Based on 6 experiments: when to use Ollama vs OpenAI, referencing actual experiment IDs]

## Evidence
[Table: model × {NDCG@5, Recall@5, MRR, latency, cost} for heading_semantic configs]

## Consequences
### Positive
- Zero API cost for embedding ($0 vs ~$0.008/run for OpenAI)
- No network dependency, data stays local
- Same 768d as mpnet — reuses FAISS index infrastructure

### Negative
- Quality delta vs OpenAI (actual numbers)
- Sequential REST calls (~50ms/text) vs batch API

### Neutral
- Hybrid retrieval can partially compensate quality gap via BM25 lexical signal
```

### Step 3.2 — Concept Library entries

**New directory:** `docs/concepts/`

**New file:** `docs/concepts/cross-encoder-reranking.md`
Sections (Notion-ready format):
- **Definition**: bi-encoder retrieval → cross-encoder reranking 2-stage pipeline
- **Why It Matters**: bi-encoder is fast but approximate; cross-encoder is slow but accurate. Combining gives best of both.
- **How It Works**: bi-encoder embeds query and doc separately (O(1) per doc at query time); cross-encoder processes (query, doc) pairs jointly (O(n)). In P5: `src/rerankers/cross_encoder.py` + `src/rerankers/cohere.py`. Reranking improved NDCG@5 by avg +0.1124 across 8 configs (Q3 data). Trade-off: ~100-200ms extra latency for +11% quality.
- **Interview Q&A** (2-3 pairs): e.g. "Why not just use cross-encoder for everything?" → O(n) per query is too slow for large corpora; bi-encoder narrows to top-K first.
- **Java/TS Parallel**: Like using Elasticsearch BM25 for candidate generation, then a BERT model for re-scoring top-50 results.

**New file:** `docs/concepts/hybrid-search.md`
Sections (Notion-ready format):
- **Definition**: combining BM25 lexical + dense vector semantic retrieval
- **Why It Matters**: neither method alone captures all relevant results. BM25 catches exact keyword matches; dense catches semantic similarity.
- **How It Works**: In P5: `src/retrievers/hybrid.py` — min-max normalize BM25 [0,∞) to [0,1], then α-weighted fusion (α=0.7 optimal). Results: hybrid (NDCG@5=0.7515) > dense (0.7176) > BM25 (0.7023) — Q2 data. RRF alternative uses ranks not scores (avoids normalization), but P5 chose score fusion for finer-grained control via α.
- **Interview Q&A** (2-3 pairs): e.g. "Why min-max normalize instead of RRF?" → score fusion preserves magnitude information; RRF only uses rank position.
- **Java/TS Parallel**: Like Elasticsearch's `bool` query combining `match` (BM25) + `knn` (vector) with boosting weights.

### Step 3.3 — Learning Journal

**New file:** `docs/learning-journal-day5.md`
- What I built: OllamaEmbedder, 6 experiment configs, Chart 11, Q5, ADR-007, concept entries
- What I learned: local vs API quality/cost trade-off (actual numbers)
- Key numbers: Ollama NDCG@5, latency, cost savings vs OpenAI
- Pattern of the day: httpx for REST API integration (vs requests: async support, better timeouts)

### Step 3.3 — Validation
```bash
ls docs/adr/ADR-007-local-vs-api-embeddings.md
ls docs/concepts/cross-encoder-reranking.md docs/concepts/hybrid-search.md
ls docs/learning-journal-day5.md
```

---

## Files Summary

### New Files (12)
| File | Purpose |
|------|---------|
| `src/embedders/ollama_embedder.py` | OllamaEmbedder class (768d, health check, httpx) |
| `experiments/configs/47_heading_ollama_dense.yaml` | Experiment config |
| `experiments/configs/48_fixed_ollama_dense.yaml` | Experiment config |
| `experiments/configs/49_sliding_ollama_dense.yaml` | Experiment config |
| `experiments/configs/50_heading_ollama_hybrid.yaml` | Experiment config |
| `experiments/configs/51_fixed_ollama_hybrid.yaml` | Experiment config |
| `experiments/configs/52_sliding_ollama_hybrid.yaml` | Experiment config |
| `tests/test_ollama_embedder.py` | Unit tests for OllamaEmbedder |
| `docs/adr/ADR-007-local-vs-api-embeddings.md` | Local vs API decision record |
| `docs/concepts/cross-encoder-reranking.md` | Concept library entry |
| `docs/concepts/hybrid-search.md` | Concept library entry |
| `docs/learning-journal-day5.md` | Learning journal |

### Modified Files (8)
| File | Changes |
|------|---------|
| `src/schemas.py:29` | Add `"ollama_nomic"` to `EmbeddingModel` Literal |
| `src/embedders/__init__.py` | Add OllamaEmbedder import + export |
| `src/factories.py:58-68` | Add `"ollama_nomic"` branch in `create_embedder()` |
| `src/experiment_runner.py:48` | Add `"ollama_nomic"` to `_EMBEDDER_ORDER` + cost constant |
| `src/visualization.py` | Add `plot_local_vs_api_comparison()` (Chart 11) |
| `results/comparison_report.md` | Append Q5 section |
| `results/iteration_log.json` | Append Ollama iteration entries |
| `pyproject.toml` | Add `httpx>=0.27.0` dependency |

---

## End-of-Session Verification Checklist

```bash
# 1. All tests green
uv run pytest tests/ -v --tb=short

# 2. Coverage ≥ 94%
uv run pytest tests/ --cov=src --cov-report=term-missing

# 3. OllamaEmbedder in factory
grep -n "ollama" src/factories.py src/schemas.py src/embedders/__init__.py

# 4. 6 YAML config files (47-52)
ls experiments/configs/4[7-9]_* experiments/configs/5[0-2]_* | wc -l  # → 6

# 5. 6 experiment result files
ls results/experiments/ | grep ollama | wc -l  # → 6

# 6. Q5 in report
grep "Q5" results/comparison_report.md

# 7. Chart 11 exists
ls results/charts/local_vs_api_comparison.png

# 8. ADR-007 exists
ls docs/adr/ADR-007-local-vs-api-embeddings.md

# 9. Concept library entries
ls docs/concepts/

# 10. Learning journal
ls docs/learning-journal-day5.md

# 11. iteration_log updated (>114 entries)
python3 -c "import json; print(len(json.load(open('results/iteration_log.json'))))"
```
