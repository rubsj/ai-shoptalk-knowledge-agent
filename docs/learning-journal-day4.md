# Learning Journal — Day 4: Experiment Execution & Analysis

## Entry 1: Deterministic Chunk IDs (Step 1)

**What:** Replaced `uuid.uuid4()` with `hashlib.md5(f"{document_id}:{start_char}:{end_char}")[:16]` for chunk IDs, and `md5(source)[:16]` for document IDs.

**Why it mattered:** Ground truth stores chunk IDs. With random UUIDs, re-chunking the same document produces different IDs every time — all metric comparisons against ground truth break silently (everything returns 0.0, but no error is raised). This would have been invisible until the experiment grid produced suspiciously identical zero scores across all configs.

**Lesson:** Any ID used for cross-run comparison must be deterministic from content/position, not random. This is the same principle behind content-addressable storage (Git blob hashes, Docker image digests). When building evaluation pipelines, trace the ID chain end-to-end: who generates the ID → who stores it → who compares against it. If any link in that chain is non-deterministic, evaluation results are meaningless.

---

## Entry 2: Cross-Chunker Ground Truth Matching (Step 3)

**What:** Ground truth was going to be generated using RecursiveChunker, but experiments run ALL 5 chunkers. Different chunkers produce different char offsets → different deterministic IDs. Exact ID matching would give 0.0 metrics for 4 out of 5 chunking strategies, making recursive appear to dominate.

**Solution:** `compute_overlap_relevance()` — instead of comparing chunk IDs, compare character offset ranges. A retrieved chunk "matches" a ground truth chunk if they overlap by ≥30% of the shorter span within the same document. Falls back to exact ID matching for legacy ground truth without offset fields.

**Why it's subtle:** This bug would have produced plausible-looking results. RecursiveChunker would score well (it generated the ground truth), and all other chunkers would score near zero. A naive analysis would conclude "recursive chunking is by far the best strategy" — which is circular reasoning, not a finding. The metrics code never errors; it just computes the correct answer to the wrong question.

**Lesson:** When building evaluation frameworks, always ask: "What assumptions does my metric computation make about the data it receives?" Here, the assumption was "ground truth chunk IDs match retrieved chunk IDs," which is only true when the chunking strategy is identical. Evaluation metrics can be perfectly correct in implementation but fundamentally broken in methodology.

---

## Entry 3: PDF Image Extraction — The Three-Iteration Journey (Step 0b)

**Iteration 1: Text-only extraction**
PyMuPDF's `page.get_text()` extracts running text but drops all figures, tables, and diagrams. For academic papers, this means the Transformer architecture diagram, BERT pre-training figures, and result tables are invisible to the RAG system. Initial extraction: 215,978 chars.

**Iteration 2: Vision LLM with append-at-end**
Added GPT-4o-mini vision: render each page as PNG at 150 DPI, send to vision model, append `[Visual Content — Page N]` description to page text. Worked — but image descriptions appeared at the bottom of each page regardless of where the figure sat. If a figure is between paragraphs 2 and 3, the description lands after paragraph 5. When chunked, the description ends up in a different chunk than the text discussing the figure.

**Iteration 3: Positional interleaving via dict-mode bounding boxes**
Used `page.get_text("dict")` to get both text blocks and image blocks with bounding boxes. Sort by y-coordinate, insert descriptions at image block positions.

**Key debugging moment:** First attempt used `get_text("blocks")` (simpler tuple format). It returned zero image blocks for a page with a known large figure. Spent time investigating — turns out `get_text("blocks")` silently omits XObject images. Only `get_text("dict")` (structured dict format) reliably returns image blocks with `type=1`. This is not documented prominently in PyMuPDF's docs. Empirical testing was essential.

**Lesson:** Library APIs that return "all the data" often have silent omissions depending on the format/mode you request. When processing PDFs, always cross-check with multiple extraction methods (`get_text()`, `get_text("blocks")`, `get_text("dict")`, `get_images()`) to understand what each one actually returns.

---

## Entry 4: Cache Pattern for Expensive Extractions

**What:** Vision LLM calls cost ~$0.01 per run and take ~60s for 61 pages. Implemented a disk cache: `save_document()` serializes to JSON, `load_document()` deserializes, `extract_all_pdfs()` checks cache first.

**Design choice:** JSON over pickle — human-readable, versionable in git, and Pydantic's `model_dump(mode="json")` / `model_validate()` handles serialization cleanly.

**Design choice:** Committed cache + source PDFs to repo. Normally you gitignore generated artifacts, but here the cache validity depends on specific PDFs AND specific vision model outputs. Committing both ensures anyone cloning the repo gets identical extraction results without needing API keys or re-running extraction.

**Lesson:** When your pipeline includes expensive external API calls, the cache boundary should be drawn right after the API call, not at the end of the pipeline. Cache the raw API output (extracted Document with descriptions), not the final processed result (chunks, embeddings). This lets you iterate on downstream processing (chunking strategies, cleaning heuristics) without re-incurring API costs.

---

## Entry 5: File Organization — Validation Subfolder

**What:** Initially saved `.txt` files alongside `.json` cache files in `data/extracted/`. Moved `.txt` to `data/extracted/validation/` for clarity.

**Why:** The `.json` files are machine-consumable cache (canonical source of truth). The `.txt` files exist purely for human inspection. Mixing them in one directory made it unclear which files were "real" artifacts vs. inspection helpers.

**Lesson:** Separate machine artifacts from human artifacts. When you have both generated data and its human-readable representation, put them in clearly separated locations. This prevents accidental modification of canonical data and makes the purpose of each file self-evident from its path.

---

## Entry 6: The Ground Truth Offset Bug — A Silent Evaluation Killer

**What broke:** Every chunk produced by `RecursiveChunker` had `start_char=0`. The `end_char` was just the chunk length, not an absolute position in the document. This meant `compute_overlap_relevance()` — the cross-chunker matching function I'd just built — was comparing all chunks against the same region (the first N characters of each document). Ground truth matching was effectively random.

**Root cause:** `RecursiveChunker.chunk()` used `document.content.find(text)` to locate each chunk's position. This works for raw chunks (they're verbatim substrings of the document). But `_merge_with_overlap()` prepends the last 50 characters of the previous chunk to each subsequent chunk. The merged text — "overlap prefix + new content" — doesn't exist as a contiguous substring in the original document. `find()` returns -1, and the fallback was `start = 0`.

```python
# The bug: overlap-merged text is NOT a substring of document.content
start = document.content.find(text)  # returns -1 for chunks 1..N
if start == -1:
    start = 0  # ← every chunk gets start_char=0
```

**Why it wasn't caught earlier:** The chunker tests verified that chunks were produced and had valid metadata types, but no test asserted `document.content[start_char:end_char] == chunk.content`. The deterministic ID tests (Step 1) verified that the same input produces the same IDs — but `make_chunk_id(doc_id, 0, chunk_length)` is deterministic too, just wrong. All 493 tests passed with the bug present.

**The fix:** Track raw chunk positions *before* overlap merge. Raw chunks are genuine substrings, so `find()` works on them. Then compute merged positions:
- Chunk 0: `start = raw_positions[0]`
- Chunk i (i > 0): `start = raw_positions[i] - chunk_overlap`

Also hardened `EmbeddingSemanticChunker` with an advancing `search_start` parameter to prevent matching earlier duplicates when using `find()`.

**The deeper problem:** This is a variant of the "metric that silently returns zero" pattern from Entry 2. The evaluation pipeline has multiple layers where a subtle data bug produces valid-looking but meaningless results:
1. Chunk IDs that are deterministic but encode wrong positions
2. Ground truth references that point to position 0 instead of the real location
3. Overlap matching that compares all chunks against the document's first paragraph
4. Metrics that are mathematically correct but computed on garbage inputs

Each layer passes its own unit tests. The bug only manifests when you trace a specific chunk ID through the full pipeline and verify it against the source document — an integration-level assertion.

**Lesson:** For any system where component A produces IDs/offsets that component B consumes for comparison, add a round-trip assertion: `source[offset_start:offset_end] == extracted_content`. This is the equivalent of a foreign key constraint in a database — it verifies referential integrity between the data producer and consumer. Without it, you're building on quicksand.

The other 4 chunkers (Fixed, SlidingWindow, HeadingSemanticChunker, EmbeddingSemanticChunker) all had correct offsets because they track positions during their splitting loop, not via `find()` after the fact. The RecursiveChunker was the only one that tried to reconstruct positions after a destructive transformation (overlap merge).

---

## Entry 7: Full Grid Run — Four Failures Before Success

Running 46 experiment configs across 4 PDFs (515 chunks) required four attempts, each failing for a different reason. The debugging sequence is worth documenting because each failure was a different category of problem — hardware, process isolation, rate limiting, and environment configuration.

### Failure 1: MPS Crash (SIGSEGV exit 139)

The first run died immediately when MiniLM tried to encode 515 chunks on the Apple Silicon GPU. SentenceTransformer defaults to MPS (Metal Performance Shaders) when it detects Apple Silicon. For small batches this works fine — all unit tests pass. But encoding 515 chunks in one call triggered a segfault in the MPS backend.

```
Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
```

This is a known-ish issue: MPS support in PyTorch is still maturing, and large batch operations on certain models can hit memory alignment or buffer overflow bugs in the Metal shaders. The symptoms are GPU-level crashes, not Python exceptions — there's nothing to catch.

**Fix:** `create_embedder(embedder_name, device="cpu")` in the experiment runner. CPU encoding of 515 chunks takes ~14s on the M5 Max (vs ~2s on GPU for batches that don't crash). Acceptable for an experiment grid that runs once. The irony: 128GB of unified memory doesn't help when the GPU shader itself crashes.

**Lesson:** "Works on small data" and "works on GPU" are two different claims. Always test your GPU codepath with production-scale data before committing to a long run. MPS is not CUDA — it's younger and less battle-tested for ML workloads.

### Failure 2: Sandbox Kills tqdm Progress Workers

After fixing the MPS crash, the next run died with workers being killed by the macOS sandbox. SentenceTransformer's `encode()` method spawns progress bar workers via Python's multiprocessing module when `show_progress_bar=True` (the default). In the Claude Code sandbox environment, child process spawning is restricted — the workers get SIGKILL'd, which propagates as a crash.

This failure was confusing because the stack trace pointed to `multiprocessing/resource_tracker.py`, not to anything in the embedding code. The connection to tqdm/progress bars isn't obvious.

**Fix:** Added `show_progress_bar=False` to both `MiniLMEmbedder.embed()` and `MpnetEmbedder.embed()`. The progress bar was informational-only — removing it has zero impact on results.

**Lesson:** Libraries that "helpfully" spawn subprocesses for UX features (progress bars, logging, monitoring) can break in restricted environments. When running in sandboxes, containers, or serverless, audit your dependencies for implicit subprocess spawning. The fix is usually a single parameter, but finding it requires understanding what the library does behind the scenes.

### Failure 3: Cohere Trial Key Rate Limiting (HTTP 429)

With GPU and process issues resolved, the grid ran successfully through 37 configs — all the MiniLM, mpnet, and BM25 experiments. It crashed on config 38: the first Cohere reranking config. Cohere's trial API key is limited to 10 calls per minute. Each config runs 18 queries, so the very first reranking config exhausted the rate limit after 10 queries.

The error was a raw `TooManyRequestsError` from the Cohere SDK. The experiment runner had no retry logic, so the entire grid crashed — losing the ability to continue with the remaining 8 configs (4 Cohere + the non-Cohere configs that hadn't run yet).

**Fix (two layers):**
1. **Retry with backoff in `CohereReranker`:** 5 retries, base delay of 10 seconds with linear backoff (`delay * (attempt + 1)`). The trial key replenishes at 10 calls/min, so a 10-20s wait is usually sufficient.
2. **Grid resilience in `_run_single_config`:** Wrapped each config execution in try/except. A failing config logs the error and continues to the next one instead of crashing the entire grid. This means a rate limit crash on Cohere config #1 doesn't prevent the remaining 7 reranking configs from attempting (they'll wait and retry too).

**Lesson:** Any experiment grid that makes API calls must be resilient at two levels: the individual API call (retry with backoff) and the grid orchestration (skip-and-continue). A grid of N configs that crashes on config K wastes K-1 runs worth of compute and API cost. The grid runner should be treated like a batch job scheduler — individual job failures shouldn't halt the entire batch.

### Failure 4: TOKENIZERS_PARALLELISM Fork Warning

Even after the above fixes, running via `nohup` produced warnings that looked like errors:

```
huggingface/tokenizers: The current process just got forked,
before the parallelism layer has been initialized...
```

This is the HuggingFace tokenizers library complaining about Python's fork behavior. When a process forks after importing `tokenizers`, the child process inherits the parent's memory but not its thread state, which can cause deadlocks. Setting `TOKENIZERS_PARALLELISM=false` suppresses this by disabling the parallelism layer entirely.

Combined with `OMP_NUM_THREADS=1` (prevents OpenMP from spawning threads that conflict with the sandbox), the full invocation became:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 nohup .venv/bin/python scripts/evaluate.py --no-judge &
```

**Fix:** Environment variables set before invocation. No code changes needed.

### The Successful Run

After all four fixes, the grid completed: 45/46 configs in 48.6 minutes, ~$0.30 estimated cost. One config failed (a Cohere reranking config that exhausted all retries — the trial key's burst limit is genuinely too low for 18 queries at 10/min even with backoff). The grid runner logged the failure and continued.

Best config: `heading_semantic_openai_dense` — NDCG@5=0.896, Recall@5=1.000, MRR=0.907.

**Meta-lesson:** The progression of failures tells a story about the layers between "code that works in tests" and "code that runs at scale": unit-test-passing code → GPU hardware limits → OS process isolation → API rate limits → library environment configuration. Each layer is invisible until you hit it, and each requires a different category of fix (device selection, parameter tuning, retry logic, environment variables). Integration testing with realistic data volume would have caught failures 1-3 before the grid run.

---

## Entry 8: Visualization, Reporting, and the Iteration Log Pattern

**What:** Built 10 chart functions in `src/visualization.py`, a comparison report generator in `src/reporting.py`, and an iteration log builder in `src/iteration_log.py`. These are the analysis artifacts that turn 45 experiment results into answers to the 4 required questions (Q1-Q4).

### The Iteration Log as Automated Ablation

The iteration log (`build_iteration_log()`) finds all config pairs that differ by exactly one parameter and records the metric delta. This is essentially automated ablation analysis — instead of manually designing "change one thing" experiments, the function discovers them from the grid results. With 45 configs and 6 comparable parameters, it found 108 single-parameter comparison pairs.

The key design decision: "before" is always the lower-NDCG config, so positive deltas mean improvement. This makes the log directional — you can read it as "changing X from A to B improved NDCG@5 by +0.24" rather than just "these two differ by 0.24." Sorting by absolute delta puts the highest-impact parameter changes at the top.

One subtlety: reranking is a 2-parameter change (`use_reranking` and `reranker_type` change together), so it doesn't appear in the single-parameter iteration log. The reranking comparison in the report handles this separately by matching reranked configs to their base.

### Precision@5 Target: Unreachable by Design

The PRD target of Precision@5 > 0.60 was not met by any of the 45 configs. Best was 0.478. This isn't a bug — it's a ceiling imposed by the ground truth. With an average of ~3 relevant chunks per query and top_k=5, a perfect retriever achieves Precision@5 = 3/5 = 0.60. Many queries have only 2 gold chunks, making their theoretical max 0.40.

This is the kind of finding the experiment grid is designed to surface. The target was set before knowing the ground truth density. In practice, Precision@5 matters less than Recall@5 for RAG — you'd rather have all relevant chunks in your top-5 (even with some noise) than miss a key passage. The system achieves Recall@5 = 1.0 on its best config, meaning every relevant chunk is retrieved.

### Report Structure: 11 Sections

The comparison report answers more than just Q1-Q4. The PRD requires iteration log traceability (Section 7g), judge target checks (Section 2b), and self-evaluation answers (Section 8c). The generator handles all cases: with judge scores, without judge scores, with/without iteration log, with/without reranking data. Each section degrades gracefully when data is missing rather than crashing.

---

## Entry 9: Judge Scores — LLM Leniency and What Calibration Revealed

**What:** Ran all 46 configs with GPT-4o as 5-axis judge (Relevance, Accuracy, Completeness, Conciseness, Citation Quality). Best config scored 4.77/5.0 overall. Then manually scored 5 diverse query-answer pairs for calibration.

### The Leniency Problem

The LLM judge gave scores between 4.5 and 4.9 on almost every axis for the best configs. That's suspiciously generous. When I scored the same 5 answers myself, the human scores were generally lower — particularly on Citation Quality and Completeness, where the LLM was giving 5s to answers that cited chunks but didn't always cite the most relevant one, or that answered the question without covering edge cases mentioned in the source text.

This is the known "leniency bias" flagged in the PRD troubleshooting guide. The LLM judge tends to score high when the answer is fluent and well-structured, even if it's missing nuance. For a portfolio project, the 4.77 score still demonstrates the pipeline works — but in production I'd anchor the judge prompt with concrete examples of 3/5 and 4/5 answers, not just describe the scale endpoints.

### What the Judge Scores Actually Tell Us

The more interesting signal from the judge is *relative* rather than absolute. Across all 46 configs, the judge scores don't vary much (most are 4.0-4.8), which tells us that once you retrieve *any* reasonable set of chunks, GPT-4o-mini generates a decent answer. The real differentiation is in the retrieval metrics (NDCG@5 ranges from 0.607 to 0.896). This confirms the PRD's hypothesis: retrieval quality is the bottleneck, not generation quality.

---

## Entry 10: Reproducibility — 0% Variance and Why

**What:** Re-ran the best config (`heading_semantic_openai_dense`) and compared all 4 retrieval metrics against the original run. Result: 0.0% variance on every metric.

**Why zero variance:** Three design choices made the pipeline fully deterministic:
1. **Deterministic chunk IDs** — `make_chunk_id(doc_id, start_char, end_char)` produces identical IDs from identical input
2. **Exact FAISS search** — `IndexFlatIP` does brute-force inner product, no approximate nearest neighbor randomness
3. **LLM response caching** — `JSONCache` returns the cached response for identical prompts, eliminating LLM temperature variance

If any of these three were non-deterministic (random IDs, approximate search like IVF/HNSW, uncached LLM calls with temperature > 0), the variance would be non-zero. The 0% result validates that the pipeline is end-to-end reproducible, which is exactly what PRD 2g requires.

**Lesson:** Reproducibility isn't a property you test for at the end — it's a property you build in from the start by choosing deterministic algorithms and caching non-deterministic external calls. If I'd used `uuid4()` for chunk IDs (the Day 1 default) or approximate FAISS indices, the reproducibility check would have failed, and debugging *which* layer introduced the variance would have been painful.

---

## Day 4 Summary

**What shipped:** Full experiment grid (46 configs, 18 queries), LLM judge scoring, 10 visualization charts, comparison report (11 sections), iteration log (114 entries), reproducibility check (0% variance), judge calibration (5 pairs with human scores). 562 tests, 94% coverage.

**Key numbers:**
- Best config: `heading_semantic_openai_dense` — NDCG@5=0.896, Recall@5=1.000, MRR=0.907
- Judge overall: 4.77/5.0 (target was >4.0)
- 3/4 retrieval targets met (Precision@5 structurally limited by GT density)
- Grid wall time: 37.7 minutes with judge, ~$0.30 total API cost
- Top insight: OpenAI embeddings provide +0.24 NDCG@5 over local models; heading-semantic chunking preserves document structure best

**What I learned:** The gap between "code that passes tests" and "code that runs a full experiment grid" involves four distinct failure categories (hardware limits, process isolation, rate limiting, environment config). Evaluation metrics can be perfectly implemented but methodologically broken if assumptions about data aren't validated. And the most valuable finding from an experiment grid isn't always the best config — it's the structural insights like "Precision@5 is ceiling-limited" and "generation quality doesn't vary much across configs."
