# Day 2 Plan: Retrieval Pipeline

## Context

Day 1 is complete: schemas, ABCs, extraction, 5 chunkers — 162 tests, 97% coverage. Day 2 builds the full retrieval pipeline: everything between "chunks exist" and "cited answer returned." This is the core RAG machinery.

## Implementation Order (8 phases, 9 commits)

### Phase 1: Embedders — `src/embedders/minilm.py`, `mpnet.py`, `__init__.py`

**MiniLMEmbedder** and **MpnetEmbedder** — both extend `BaseEmbedder`:
- `__init__`: load `SentenceTransformer` eagerly (128GB, no lazy needed)
- `embed(texts)`: `model.encode(texts, convert_to_numpy=True)` → `faiss.normalize_L2()` → return
- `embed_query(query)`: delegate to `embed([query])[0]`
- `dimensions` property: 384 / 768 respectively
- Intentional duplication — two small classes > parameterized abstraction

**Tests** (`tests/test_embedders.py`): Mock `SentenceTransformer`. Verify shape, L2 normalization, dimensions, isinstance. Parametrize for both models.

**Commit:** `feat(p5): add MiniLM and mpnet embedders with L2 normalization`

---

### Phase 2: Vector Store — `src/vector_store.py`

**FAISSVectorStore(BaseVectorStore)**:
- `__init__(dimension)`: `faiss.IndexFlatIP(dimension)`, `_chunks: list[Chunk] = []`
- `add(chunks, embeddings)`: validate shapes → `.astype(float32).copy()` → `faiss.normalize_L2()` → `index.add()` → extend `_chunks`
- `search(query_embedding, top_k)`: normalize query → `index.search(query, min(top_k, ntotal))` → skip `-1` sentinels → return `list[tuple[Chunk, float]]`
- `save(path)`: `faiss.write_index()` + JSON chunks (exclude `embedding` field — not serializable, lives in FAISS binary)
- `load(path)`: `faiss.read_index()` + reconstruct `Chunk` objects from JSON
- `__len__`: `index.ntotal`

Key: `.copy()` before normalize_L2 to avoid mutating caller's array.

**Tests** (`tests/test_vector_store.py`): Real FAISS (fast on small data). add+search, empty index, dimension mismatch, save/load roundtrip via `tmp_path`, cosine score ≈ 1.0 for self-search.

**Commit:** `feat(p5): add FAISSVectorStore with persistence`

---

### Phase 3: Retrievers — `src/retrievers/dense.py`, `bm25.py`, `hybrid.py`, `__init__.py`

**DenseRetriever(BaseRetriever)**:
- `__init__(embedder, vector_store)`
- `retrieve(query, top_k)`: embed query → search → wrap in `RetrievalResult(retriever_type="dense")`

**BM25Retriever(BaseRetriever)**:
- `__init__(chunks)`: tokenize with `.lower().split()` → `BM25Okapi(tokenized)`. Guard empty corpus.
- `retrieve(query, top_k)`: tokenize query → `get_scores()` → `argsort` → top-k → `RetrievalResult(retriever_type="bm25")`

**HybridRetriever(BaseRetriever)** — PRD Decision 2 (internal composition):
- `__init__(dense_retriever, bm25_retriever, alpha=0.7)` — factory injects sub-retrievers
- `retrieve(query, top_k)`:
  1. Oversample: get `top_k * 2` from each sub-retriever
  2. Min-max normalize BM25: `(v - min) / (max - min)`. Edge case: all identical → set to 0.0
  3. Union chunks by ID: `combined = alpha * dense_score + (1-alpha) * bm25_norm`
  4. Sort descending, take top_k, `retriever_type="hybrid"`

**Tests** (`tests/test_retrievers.py`):
- Dense: mock embedder + store
- BM25: real `rank_bm25` (lightweight)
- Hybrid (critical): mock sub-retrievers with known scores. Test: normalization, identical BM25 edge case, alpha=1.0 (pure dense), alpha=0.0 (pure BM25), chunk-only-in-one-retriever

**Commit:** `feat(p5): add dense, BM25, and hybrid retrievers`

---

### Phase 4: Rerankers — `src/rerankers/cross_encoder.py`, `cohere_reranker.py`, `__init__.py`

**CrossEncoderReranker(BaseReranker)**:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `rerank(query, results, top_k)`: build (query, chunk.content) pairs → `model.predict()` → sort by score → top_k with new ranks
- Cross-encoder score replaces original score. `retriever_type` preserved (provenance).

**CohereReranker(BaseReranker)**:
- `__init__(api_key=None)`: `cohere.ClientV2(api_key or os.environ["COHERE_API_KEY"])`
- `rerank()`: `co.rerank(model="rerank-english-v3.0", ...)` → map `.index` back to original results

**Tests** (`tests/test_rerankers.py`): Mock `CrossEncoder.predict()` and `cohere.ClientV2.rerank()`. Verify reordering, top_k, sequential ranks.

**Commit:** `feat(p5): add CrossEncoder and Cohere rerankers`

---

### Phase 5: Cache — `src/cache.py`

> Moved before Generator so LiteLLMClient can depend on it.

**JSONCache**:
- `__init__(cache_dir: str = "data/cache")`: `Path(cache_dir).mkdir(parents=True, exist_ok=True)`
- `make_key(model, system_prompt, user_prompt)` → MD5 hash (matches CLAUDE.md format: `f"{model}\n{system_prompt}\n---\n{user_prompt}"`)
- `get(key)` → load `{key}.json` or return None
- `set(key, value)` → write `{key}.json`

**Tests** (`tests/test_cache.py`): roundtrip, miss returns None, deterministic keys, creates missing dirs.

**Commit:** `feat(p5): add JSON cache for LLM responses`

---

### Phase 6: Generator + Citations — `src/generator.py`

> Depends on Phase 5 (Cache). LiteLLMClient accepts optional cache to avoid redundant API calls.

**LiteLLMClient(BaseLLM)**:
- `__init__(model: str = "gpt-4o-mini", cache: JSONCache | None = None)`: store model and cache
- `generate(prompt, system_prompt, temperature)`:
  1. Build cache key via `cache.make_key(model, system_prompt, prompt)` if cache provided
  2. Check `cache.get(key)` — if hit, return cached content string
  3. Call `litellm.completion()` → extract `.choices[0].message.content`
  4. `cache.set(key, {"content": content})` if cache provided
  5. Return content
- Omit system message from LiteLLM messages list if empty string

**build_qa_prompt(query, chunks) -> str**: Number chunks [1]...[N], instruct LLM to cite with [N].

**extract_citations(answer, chunks) -> list[Citation]**:
- Regex `r'\[(\d+)\]'`, validate 1 ≤ N ≤ len(chunks), deduplicate
- `relevance_score=0.0` (set later by judge on Day 3)
- `text_snippet=chunk.content[:100]`
- Out-of-range: warn + skip

**Tests** (`tests/test_generator.py`):
- Mock `litellm.completion`. Test prompt building, citation parsing edge cases (valid, out-of-range, duplicates, no markers, [0]).
- **Cache integration tests:**
  - `test_generate_uses_cache_on_hit` — pre-populate cache, verify `litellm.completion` NOT called
  - `test_generate_populates_cache_on_miss` — verify `cache.set()` called after successful completion
  - `test_generate_works_without_cache` — `cache=None`, verify no errors

**Commit:** `feat(p5): add LiteLLM generator with citation extraction and cache integration`

---

### Phase 7: Factory — `src/factories.py`

Factory functions map config strings → class instances:

```python
def create_chunker(config: ExperimentConfig) -> BaseChunker
def create_embedder(model_name: str) -> BaseEmbedder
def create_retriever(config, embedder, chunks, vector_store) -> BaseRetriever
def create_reranker(reranker_type: str) -> BaseReranker
def create_llm(model: str = "gpt-4o-mini", cache: JSONCache | None = None) -> BaseLLM
```

Chunker mapping (verified against actual Day 1 `__init__` signatures):
- `"fixed"` → `FixedSizeChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)`
  - Actual signature: `__init__(self, chunk_size: int = 512, chunk_overlap: int = 50)`
- `"recursive"` → `RecursiveChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)`
  - Actual signature: `__init__(self, chunk_size: int = 512, chunk_overlap: int = 50, separators: list[str] | None = None)`
- `"sliding_window"` → `SlidingWindowChunker(window_size=config.window_size_tokens, step_size=config.step_size_tokens)`
  - Actual signature: `__init__(self, window_size: int = 200, step_size: int = 150, encoding_name: str = "cl100k_base")`
  - **NOTE:** ExperimentConfig fields are `window_size_tokens` / `step_size_tokens` but chunker params are `window_size` / `step_size` — factory maps between names
- `"heading_semantic"` → `HeadingSemanticChunker(min_chunk_size=config.min_chunk_size)`
  - Actual signature: `__init__(self, heading_patterns: list[re.Pattern] | None = None, min_chunk_size: int = 50, max_chunk_size: int = 3000)`
  - Factory passes `min_chunk_size` only; `heading_patterns` and `max_chunk_size` use defaults
- `"embedding_semantic"` → `EmbeddingSemanticChunker(breakpoint_threshold=config.breakpoint_threshold, min_chunk_size=config.min_chunk_size)`
  - Actual signature: `__init__(self, breakpoint_threshold: float = 0.85, min_chunk_size: int = 100)`

Retriever logic:
- `"bm25"` → `BM25Retriever(chunks)`
- `"dense"` → `DenseRetriever(embedder, vector_store)`
- `"hybrid"` → create DenseRetriever + BM25Retriever → `HybridRetriever(dense, bm25, alpha)`

**Tests** (`tests/test_factories.py`): Mock heavyweight constructors. Verify correct types returned. Test unknown strings raise.

**Commit:** `feat(p5): add factory functions for config-to-class mapping`

---

### Phase 8: Smoke Test + ADRs

**Smoke test** (`tests/test_smoke.py`):
End-to-end with mocked LLM/models: Document → FixedSizeChunker → MiniLMEmbedder (mocked) → FAISSVectorStore → HybridRetriever → CrossEncoderReranker (mocked) → LiteLLMClient (mocked with cache, returns "[1] and [2]") → extract_citations → verify 2 citations with valid chunk IDs.

**Cache verification in smoke test:**
- Run the same query twice through the generator with cache enabled
- Assert `litellm.completion` mock is called exactly once (second call hits cache)
- Assert both calls return identical answers

**ADR-003** (`docs/adr/ADR-003-hybrid-retrieval-with-min-max-score-fusion.md`):
BM25 [0,∞) vs cosine [0,1]. Min-max normalize. Edge case: identical scores → 0.0. Alpha default 0.7.

**ADR-004** (`docs/adr/ADR-004-litellm-over-raw-openai-sdk.md`):
Multi-provider wrapper. Model string routing. Alternatives: raw OpenAI (locks provider), LangChain (banned).

**Commits:**
- `test(p5): add end-to-end smoke test with cache verification`
- `docs(p5): add ADR-003 and ADR-004`

---

## Files Modified/Created (complete list)

| File | Action |
|------|--------|
| `src/embedders/minilm.py` | Implement |
| `src/embedders/mpnet.py` | Implement |
| `src/embedders/__init__.py` | Update exports |
| `src/vector_store.py` | Implement |
| `src/retrievers/dense.py` | Implement |
| `src/retrievers/bm25.py` | Implement |
| `src/retrievers/hybrid.py` | Implement |
| `src/retrievers/__init__.py` | Update exports |
| `src/rerankers/cross_encoder.py` | Implement |
| `src/rerankers/cohere_reranker.py` | Implement |
| `src/rerankers/__init__.py` | Update exports |
| `src/cache.py` | Implement |
| `src/generator.py` | Implement |
| `src/factories.py` | Implement |
| `tests/test_embedders.py` | Create |
| `tests/test_vector_store.py` | Create |
| `tests/test_retrievers.py` | Create |
| `tests/test_rerankers.py` | Create |
| `tests/test_cache.py` | Create |
| `tests/test_generator.py` | Create |
| `tests/test_factories.py` | Create |
| `tests/test_smoke.py` | Create |
| `docs/adr/ADR-003-hybrid-retrieval-with-min-max-score-fusion.md` | Create |
| `docs/adr/ADR-004-litellm-over-raw-openai-sdk.md` | Create |

## Pitfalls to Watch

1. **FAISS normalize_L2 needs contiguous float32** — always `.astype(np.float32).copy()` before
2. **BM25Okapi empty corpus** — guard with `if not chunks` check
3. **Cohere SDK v5** — `cohere.ClientV2`, response has `.results[i].index` and `.relevance_score`
4. **Citation `relevance_score`** — schema requires `[0,1]`, set 0.0 at extraction time
5. **Chunk serialization** — exclude `embedding` (ndarray) from JSON in save/load
6. **SlidingWindowChunker param name mismatch** — ExperimentConfig uses `window_size_tokens`/`step_size_tokens`, chunker uses `window_size`/`step_size`. Factory must map between them.
7. **HeadingSemanticChunker constructor** — takes `(heading_patterns, min_chunk_size, max_chunk_size)`, NOT `(chunk_size, chunk_overlap)`

## Verification

After each commit:
```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

Day 2 is done when:
1. All tests green, ≥95% coverage on every new module
2. Total tests ≈ 230+ (162 Day 1 + ~70 Day 2)
3. Smoke test passes: full pipeline wires together, cache deduplicates LLM calls
4. `uv run python -c "..."` can load PDF → chunk → embed → index → retrieve → generate → cited answer

## Branch

`feat/p5-day2-retrieval-pipeline` — single branch, 9 sequential commits
