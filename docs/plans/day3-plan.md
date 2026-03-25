# Day 3 Implementation Plan: Evaluation Framework

## Context

Day 2 is complete and merged to main (HEAD: 7c930f0). 295 tests passing, >=95% coverage. Full retrieval pipeline works end-to-end. Day 3 builds the evaluation framework: hand-rolled IR metrics, ground truth generation, 5-axis LLM-as-Judge, OpenAI embedder, 35+ YAML experiment configs, and the experiment runner that ties everything together.

Stub files already exist (docstrings only): `src/evaluation/metrics.py`, `src/evaluation/judge.py`, `src/evaluation/ground_truth.py`, `src/embedders/openai_embedder.py`. Pydantic schemas for `ExperimentConfig`, `ExperimentResult`, `RetrievalMetrics`, `JudgeScores`, `PerformanceMetrics` are already in `src/schemas.py`.

---

## Implementation Order (11 steps, dependency-first)

### Step 1: `src/schemas.py` — Add 6 Pydantic models

Add after `ExperimentResult`:

```python
class GroundTruthChunk(BaseModel):
    """Single chunk relevance judgment."""
    chunk_id: str = Field(..., min_length=1)
    relevance_grade: int = Field(..., ge=0, le=3)
    # 0=irrelevant, 1=same doc, 2=same section, 3=gold

class GroundTruthQuery(BaseModel):
    """One evaluation query with gold chunks."""
    query_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    relevant_chunks: list[GroundTruthChunk] = Field(..., min_length=1)

class GroundTruthSet(BaseModel):
    """Full ground truth dataset."""
    queries: list[GroundTruthQuery] = Field(..., min_length=1)

class JudgeResult(BaseModel):
    """Single answer score from 5-axis LLM-as-Judge (Instructor output)."""
    relevance: int = Field(..., ge=1, le=5)
    accuracy: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    conciseness: int = Field(..., ge=1, le=5)
    citation_quality: int = Field(..., ge=1, le=5)

class GeneratedQAPair(BaseModel):
    """Instructor response model for ground truth generation."""
    question: str = Field(..., min_length=1)
    relevant_chunks: list[GroundTruthChunk] = Field(..., min_length=1)

class QueryResult(BaseModel):
    """Per-query result for difficulty analysis and aggregation."""
    query_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_scores: RetrievalMetrics = Field(..., description="Per-query retrieval metrics")
    judge_result: JudgeResult | None = Field(default=None)
    latency_ms: float = Field(..., ge=0.0)
```

Also add two fields to the existing `PerformanceMetrics` model:

```python
# Add to PerformanceMetrics
embedding_source: str = Field(..., description="'local', 'api', or 'none' (BM25)")
cost_estimate_usd: float = Field(default=0.0, ge=0.0, description="Estimated API cost in USD")
```

And update `ExperimentResult.query_results` type from `list[dict]` to `list[QueryResult]`.

**Tests** (add to `tests/test_schemas.py`): Reject `relevance_grade` outside 0-3, reject empty `relevant_chunks`, reject scores outside 1-5, valid construction for each. Test `QueryResult` with and without `judge_result`. Test `PerformanceMetrics` with `embedding_source` values. Test `GeneratedQAPair` valid construction.

---

### Step 2: `src/evaluation/metrics.py` — IR metrics from scratch

```python
from __future__ import annotations
import math

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """% of top-K that are relevant. Returns 0.0 if k <= 0."""

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """% of relevant found in top-K. Returns 0.0 if relevant_ids empty."""

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of first relevant result. Returns 0.0 if none found."""

def ndcg_at_k(retrieved_ids: list[str], graded_relevance: dict[str, int], k: int) -> float:
    """NDCG with 4-level grading. DCG = sum(rel_i / log2(i+2)). NDCG = DCG/ideal_DCG."""
```

**Key logic for NDCG**:
- Internal `_dcg(ids, relevance, k)`: sum `relevance.get(id, 0) / math.log2(i + 2)` for `i` in `range(min(k, len(ids)))`
- Ideal DCG: sort all relevance values descending, take top-k, compute DCG on ideal ordering
- Return 0.0 if ideal_DCG == 0.0

**Tests** (`tests/test_metrics.py`):
- `precision_at_k`: all relevant, none relevant, partial, k > len(retrieved), k=0
- `recall_at_k`: full, partial, empty relevant returns 0.0
- `mrr`: rank 1 -> 1.0, rank 3 -> 1/3, no hit -> 0.0
- `ndcg_at_k`: perfect ranking -> 1.0, reversed -> <1.0, all irrelevant -> 0.0, hand-computed graded example {3,2,1,0}
- Edge: empty lists, k=0, single element

---

### Step 3: `src/evaluation/ground_truth.py` — Loading + generation

```python
from __future__ import annotations
import json
import logging
from pathlib import Path
import instructor
import litellm
from src.schemas import Chunk, GeneratedQAPair, GroundTruthSet

def load_ground_truth(path: str) -> GroundTruthSet:
    """Read JSON, validate through GroundTruthSet. Pydantic catches bad data."""

def generate_ground_truth_candidates(
    chunks: list[Chunk], n: int = 30, model: str = "gpt-4o",
) -> list[GeneratedQAPair]:
    """LLM generates n QA pairs from chunks via Instructor. Returns validated models for human curation."""
```

**Key logic for `generate_ground_truth_candidates`**:
- Create patched client: `client = instructor.from_litellm(litellm.completion)`
- Batch chunks into groups of ~10. For each batch, build a prompt that includes all chunk IDs and their content (e.g., `[chunk_abc123]: "text..."`) so the LLM can reference them by ID
- The LLM generates one question answerable by 1-3 chunks from that batch, assigns relevance grades to each chunk in the batch (grade 3 for gold, 2 for same-section context, 1 for same-doc, 0 for irrelevant). This produces multi-grade ground truth without requiring the developer to manually assign grade-2 and grade-1 chunks during curation
- For each batch, call `client.chat.completions.create(response_model=GeneratedQAPair, max_retries=3, messages=[...])`
- Instructor handles JSON parsing, Pydantic validation, and automatic retry on failure
- `GeneratedQAPair.relevant_chunks` references chunk_ids from the batch (only chunks with grade >= 1)
- Return `list[GeneratedQAPair]` (validated models for human curation, not raw dicts)

**Tests** (`tests/test_ground_truth.py`):
- `load_ground_truth`: valid JSON loads, missing field raises `ValidationError`, file not found raises
- `generate_ground_truth_candidates`: mock `instructor.from_litellm` and patched client, verify returns `list[GeneratedQAPair]`, handles empty chunks

---

### Step 4: `src/evaluation/judge.py` — 5-axis LLM-as-Judge

```python
from __future__ import annotations
import logging
import instructor
import litellm
from src.cache import JSONCache
from src.schemas import Chunk, JudgeResult, JudgeScores

_JUDGE_SYSTEM_PROMPT = """You are an expert RAG evaluation judge. Score the answer on 5 axes (1-5 each).

Rubric:
  Relevance:  1=off-topic, 3=partially addresses, 5=directly answers
  Accuracy:   1=major errors, 3=minor errors, 5=every claim verifiable
  Completeness: 1=fragment, 3=covers basics, 5=comprehensive
  Conciseness: 1=extremely verbose, 3=some filler, 5=focused
  Citation Quality: 1=none/wrong, 3=some correct, 5=every claim cited"""

class LLMJudge:
    def __init__(self, model: str = "gpt-4o", cache: JSONCache | None = None) -> None:
        """Create Instructor-patched client via instructor.from_litellm(litellm.completion)."""
    def score(self, query: str, answer: str, chunks: list[Chunk]) -> JudgeResult:
        """Score one answer via Instructor. response_model=JudgeResult, max_retries=3. Cache via JSONCache."""
    def score_batch(self, qa_pairs: list[dict]) -> JudgeScores:
        """Score all pairs, compute per-axis averages, return JudgeScores."""
```

**Key logic**:
- `__init__`: `self._client = instructor.from_litellm(litellm.completion)`, store `self._model` and `self._cache`
- `score`: build user prompt with query + answer + numbered chunks. Check cache first (key = `cache.make_key("judge", _JUDGE_SYSTEM_PROMPT, user_prompt)`). On cache hit, reconstruct with `JudgeResult.model_validate(cached_data)`. On miss, call `self._client.chat.completions.create(model=self._model, response_model=JudgeResult, max_retries=3, messages=[...])`. Instructor handles JSON parsing, Pydantic validation, and retry. Cache with `self._cache.set(key, result.model_dump())` — serialize Pydantic to dict for JSON storage. Same `model_dump()`/`model_validate()` pattern applies anywhere Pydantic models are cached.
- `score_batch`: call `score()` per pair, collect `JudgeResult` list, average each axis, compute `overall_average = mean(5 axis avgs)`

**Tests** (`tests/test_judge.py`):
- Mock `instructor.from_litellm` and patched client -> correct JudgeResult
- Cache hit -> client not called twice
- `score_batch`: averages computed correctly over multiple results
- `overall_average` = mean of 5 axes

---

### Step 5: `src/evaluation/__init__.py` — Wire exports

```python
from src.evaluation.ground_truth import generate_ground_truth_candidates, load_ground_truth
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
```

---

### Step 6: `src/embedders/openai_embedder.py` — text-embedding-3-small

```python
from __future__ import annotations
import faiss
import numpy as np
import litellm
from src.interfaces import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """1536d API-based embedder. Zero local RAM. Runs LAST in grid."""
    _MODEL_NAME = "text-embedding-3-small"
    _DIMENSIONS = 1536
    _BATCH_SIZE = 100

    def __init__(self) -> None: pass  # API-based, nothing to load
    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch via litellm.embedding(). L2-normalize with faiss.normalize_L2()."""
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
    @property
    def dimensions(self) -> int: return self._DIMENSIONS
```

**Key logic for `embed`**:
- Empty list -> `np.empty((0, 1536), dtype=np.float32)`
- Batch loop: `litellm.embedding(model=self._MODEL_NAME, input=batch)` per `_BATCH_SIZE`
- Extract: `[item["embedding"] for item in response.data]`
- Stack, cast to float32, `np.ascontiguousarray()`, `faiss.normalize_L2()`

**Tests** (add `TestOpenAIEmbedder` class to `tests/test_embedders.py`):
- Mock `litellm.embedding`, verify shape `(N, 1536)`, L2-norm, empty list, `embed_query` 1D, `dimensions=1536`
- 150 texts -> 2 API calls (100+50)

---

### Step 7: Update `src/embedders/__init__.py` + `src/factories.py`

**embedders/__init__.py**: Add `OpenAIEmbedder` to exports.

**factories.py**:
1. Add `"openai"` branch to `create_embedder()` (lazy import like `create_llm`)
2. Add new function:

```python
def load_configs(config_dir: str = "experiments/configs") -> list[ExperimentConfig]:
    """Load all YAML configs from directory. yaml.safe_load() ONLY."""
    import yaml
    from pathlib import Path
    configs = []
    for yaml_file in sorted(Path(config_dir).glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text())
        configs.append(ExperimentConfig.model_validate(data))
    return configs
```

**Tests** (add to `tests/test_factories.py`):
- `create_embedder("openai")` returns `OpenAIEmbedder` instance (mock litellm)
- `load_configs`: valid YAMLs load, invalid raises, empty dir -> empty list, sorted alphabetically

---

### Step 8: `experiments/configs/*.yaml` — 35 experiment configs

**Naming**: `{NN}_{chunking_strategy}_{embedder}_{retriever}.yaml`

**Grid** (35 files):
| Range | Configs | Embedder | Retriever |
|-------|---------|----------|-----------|
| 01-05 | 5 chunkers | minilm | dense |
| 06-10 | 5 chunkers | mpnet | dense |
| 11-15 | 5 chunkers | minilm | hybrid (alpha=0.7) |
| 16-20 | 5 chunkers | mpnet | hybrid (alpha=0.7) |
| 21-25 | 5 chunkers | (none) | bm25 |
| 26-30 | 5 chunkers | openai | dense |
| 31-35 | 5 chunkers | openai | hybrid (alpha=0.7) |

All use: `chunk_size=512, chunk_overlap=50, top_k=5, use_reranking=false`.
Sliding window configs add: `window_size_tokens=128, step_size_tokens=64`.
BM25 configs: `embedding_model: null`.

**Tests** (`tests/test_configs.py`):
- Load all YAMLs, validate with `ExperimentConfig`, assert count >= 35
- BM25 configs have `embedding_model: None`
- Hybrid configs have `hybrid_alpha` set
- Sliding window configs have `window_size_tokens` + `step_size_tokens`

---

### Step 9: `src/experiment_runner.py` — Experiment grid orchestrator (stub exists)

```python
from __future__ import annotations
import gc, json, logging, os, time, uuid
from pathlib import Path
import psutil

_EMBEDDER_ORDER: list[str | None] = ["minilm", "mpnet", None, "openai"]

def _group_configs_by_embedder(
    configs: list[ExperimentConfig],
) -> dict[str | None, list[ExperimentConfig]]:
    """Group by embedding_model. None = BM25-only."""

def _run_single_config(
    config: ExperimentConfig,
    embedder: BaseEmbedder | None,
    documents: list[Document],
    ground_truth: GroundTruthSet,
    judge: LLMJudge | None,
    cache: JSONCache | None,
) -> ExperimentResult:
    """One config: chunk -> embed -> index -> retrieve -> evaluate -> judge -> perf metrics."""

def run_experiment_grid(
    config_dir: str = "experiments/configs",
    ground_truth_path: str = "data/ground_truth.json",
    output_dir: str = "results/experiments",
    documents: list[Document] | None = None,
    run_judge: bool = True,
) -> list[ExperimentResult]:
    """Main entry. Groups by embedder, pools model loading, saves per-experiment JSON + summary."""
```

**Pooling algorithm**:
```
1. Load configs + ground truth
2. Create cache + judge (gpt-4o for eval quality)
3. Group configs by embedder
4. For each embedder in _EMBEDDER_ORDER:
     Load model ONCE -> run all configs in group -> del model + gc.collect()
5. Save each ExperimentResult as results/experiments/{experiment_id}.json
6. Save results/experiments/summary.json with all results for easy loading
```

**_run_single_config flow**:
```
1. Track time + memory (psutil)
2. Chunk document(s)
3. Embed chunks + build FAISS index (skip if BM25)
4. Create retriever
5. For each ground truth query:
   a. Retrieve top-K
   b. Generate answer via LLM
   c. Extract citations
   d. Record latency
   e. Compute per-query retrieval metrics (precision, recall, MRR, NDCG)
   f. Run judge on this query's answer (if enabled)
   g. Build QueryResult with per-query metrics + judge_result
6. Aggregate retrieval metrics (average per-query RetrievalMetrics)
7. Aggregate judge scores (average per-query JudgeResult into JudgeScores)
8. Measure index size (temp save + stat)
9. Set embedding_source: "local" for minilm/mpnet, "api" for openai, "none" for BM25
10. Compute cost_estimate_usd for API embedders (~$0.02/1M tokens, estimate from chunk count x avg tokens)
11. Return ExperimentResult with query_results: list[QueryResult]
```

**Tests** (`tests/test_experiment_runner.py`):
- `_group_configs_by_embedder`: correct grouping
- `_run_single_config`: mock everything, verify ExperimentResult structure, verify `query_results` contains `QueryResult` objects with per-query metrics
- BM25 config runs without embedder, `embedding_source="none"`, `cost_estimate_usd=0.0`
- OpenAI config sets `embedding_source="api"` with non-zero `cost_estimate_usd`
- `run_experiment_grid`: verify pooling order, embedder loaded once per group, `gc.collect()` called between groups
- Per-experiment JSON files saved to `results/experiments/`
- `summary.json` written with all results

---

### Step 10: `data/ground_truth.json` — Placeholder

```json
{
  "queries": []
}
```

Empty array won't pass `GroundTruthSet` validation (min_length=1) -- intentional. Forces curation before experiments. All runner tests mock ground truth loading.

---

### Step 11: ADR-005 — YAML + Pydantic for experiment configs

Write to `docs/adr/ADR-005-yaml-pydantic-experiment-configs.md` following existing ADR template. Key points: `yaml.safe_load()` only, Pydantic cross-field validators catch invalid combos, naming convention controls execution order, factory pattern maps YAML strings to classes.

---

## Files Modified/Created Summary

| File | Action | ~LOC |
|------|--------|------|
| `src/schemas.py` | MODIFY (+6 models, +2 fields) | +70 |
| `src/evaluation/metrics.py` | IMPLEMENT | ~80 |
| `src/evaluation/ground_truth.py` | IMPLEMENT (Instructor) | ~80 |
| `src/evaluation/judge.py` | IMPLEMENT (Instructor) | ~100 |
| `src/evaluation/__init__.py` | UPDATE exports | ~12 |
| `src/embedders/openai_embedder.py` | IMPLEMENT | ~55 |
| `src/embedders/__init__.py` | UPDATE exports | +2 |
| `src/factories.py` | UPDATE (+openai, +load_configs) | +30 |
| `experiments/configs/*.yaml` | CREATE 35 files | ~280 |
| `src/experiment_runner.py` | IMPLEMENT (stub exists) | ~220 |
| `data/ground_truth.json` | CREATE placeholder | ~3 |
| `docs/adr/ADR-005-*.md` | CREATE | ~50 |
| `tests/test_schemas.py` | MODIFY | +60 |
| `tests/test_metrics.py` | CREATE | ~150 |
| `tests/test_ground_truth.py` | CREATE | ~120 |
| `tests/test_judge.py` | CREATE | ~150 |
| `tests/test_embedders.py` | MODIFY | +80 |
| `tests/test_factories.py` | MODIFY | +60 |
| `tests/test_configs.py` | CREATE | ~50 |
| `tests/test_experiment_runner.py` | CREATE | ~220 |
| **Total** | | **~1,900** |

---

## Verification

1. `pytest tests/ -v --cov=src --cov-report=term-missing` — all pass, >=95% per module
2. `python -c "from src.evaluation.metrics import ndcg_at_k; print(ndcg_at_k(['a','b','c'], {'a':3,'b':0,'c':1}, k=3))"` — verify hand-computed NDCG
3. `python -c "from src.factories import load_configs; configs = load_configs('experiments/configs'); print(len(configs))"` — prints >= 35
4. `python -c "from src.evaluation.ground_truth import load_ground_truth"` — imports clean
5. Verify each YAML individually: `python -c "import yaml; from src.schemas import ExperimentConfig; ExperimentConfig.model_validate(yaml.safe_load(open('experiments/configs/01_fixed_minilm_dense.yaml')))"`
6. ADR-005 exists at `docs/adr/ADR-005-yaml-pydantic-experiment-configs.md`

## Execution Order for Sonnet

1. `src/schemas.py` + `tests/test_schemas.py` additions -> commit
2. `src/evaluation/metrics.py` + `tests/test_metrics.py` -> commit
3. `src/evaluation/ground_truth.py` + `tests/test_ground_truth.py` -> commit
4. `src/evaluation/judge.py` + `tests/test_judge.py` -> commit
5. `src/evaluation/__init__.py` -> commit with steps 3-4
6. `src/embedders/openai_embedder.py` + `src/embedders/__init__.py` + embedder tests -> commit
7. `src/factories.py` updates + factory tests -> commit
8. `experiments/configs/*.yaml` + `tests/test_configs.py` -> commit
9. `src/experiment_runner.py` + `tests/test_experiment_runner.py` -> commit
10. `data/ground_truth.json` placeholder -> commit with step 9
11. `docs/adr/ADR-005-*.md` -> commit
