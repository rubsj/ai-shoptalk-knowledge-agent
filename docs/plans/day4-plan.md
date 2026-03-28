# Day 4 Plan — Experiment Execution & Analysis

## Context
Day 3 is complete (d1d76a5, 484 tests, 95%+ coverage). All infrastructure is built: 5 chunkers, 3 embedders, 3 retrievers, 2 rerankers, LLM judge, IR metrics, experiment runner, 35 YAML configs. But `data/ground_truth.json` is empty (`{"queries": []}`), `src/visualization.py` is a 16-line stub, no reranking configs exist, and chunk IDs are non-deterministic (uuid4) — meaning ground truth chunk references would break on re-chunking. Day 4 populates ground truth, runs the full grid, and produces all analysis artifacts.

---

## Critical Blocker: Deterministic Chunk IDs

**Problem:** `Chunk.id = uuid.uuid4()` means every re-chunking produces new IDs. Ground truth references chunk IDs by value. Metrics compare `retrieved_chunk_ids` against `gt_query.relevant_chunks[].chunk_id` — all comparisons fail if IDs change.

**Fix:**
1. Add `make_chunk_id(document_id, start_char, end_char) -> str` to `src/chunkers/_utils.py` — uses `hashlib.md5(f"{document_id}:{start_char}:{end_char}")[:16]`
2. Make `Document.id` deterministic: hash of `metadata.source` (file path) instead of uuid4
3. Update all 5 chunkers to call `make_chunk_id()` instead of `uuid.uuid4()`
4. Update chunker tests to verify same input → same IDs

**Files:** `src/schemas.py` (Document.id), `src/chunkers/_utils.py`, `src/chunkers/fixed.py`, `recursive.py`, `sliding_window.py`, `heading_semantic.py`, `embedding_semantic.py`, tests

---

## Critical Blocker: Ground Truth Cross-Chunker Matching (FIX 1)

**Problem:** Ground truth is generated using RecursiveChunker(512, 50) and stores `chunk_id` references. But the experiment grid runs ALL 5 chunkers — each produces DIFFERENT chunks at DIFFERENT char offsets → DIFFERENT deterministic IDs. The metrics code in `_run_single_config` (lines 124-135) does exact ID comparison:
```python
relevant_ids = {c.chunk_id for c in gt_query.relevant_chunks if c.relevance_grade >= 1}
# ... then ...
recall_at_k(retrieved_ids, relevant_ids, k=5)
```
For any config NOT using RecursiveChunker(512, 50), `retrieved_ids ∩ relevant_ids = ∅` and all retrieval metrics silently = 0.0. This makes recursive chunker appear to dominate — not because it's better, but because it's the only strategy whose IDs match ground truth.

**Chosen approach: Option B — Document-section matching via char offset overlap.**

Why Option B:
- `ChunkMetadata` already stores `document_id`, `start_char`, `end_char` — no new extraction needed
- Deterministic and fast (numeric range comparison, not text similarity)
- Works across all chunking strategies without re-mapping
- Avoids the complexity of Option C (re-chunk per strategy) and the fuzziness of Option A (text overlap)

**Implementation:**

1. **Extend `GroundTruthChunk` schema** (`src/schemas.py`):
   ```python
   class GroundTruthChunk(BaseModel):
       chunk_id: str  # kept for backward compat / logging
       document_id: str  # NEW — which document
       start_char: int   # NEW — section start in document.content
       end_char: int      # NEW — section end in document.content
       relevance_grade: int  # 0-3
   ```

2. **Add `compute_overlap_relevance()` to `src/evaluation/metrics.py`**:
   ```python
   def compute_overlap_relevance(
       retrieved_chunks: list[Chunk],
       gt_chunks: list[GroundTruthChunk],
       min_overlap_ratio: float = 0.3,
   ) -> tuple[set[str], dict[str, int]]:
       """Map retrieved chunk IDs to ground truth by char offset overlap.

       A retrieved chunk 'hits' a ground truth section if:
         overlap_chars / min(len(retrieved), len(gt_section)) >= min_overlap_ratio

       Returns:
           relevant_ids: set of retrieved chunk IDs that overlap with any GT section (grade >= 1)
           graded_relevance: dict mapping retrieved chunk ID → highest matching GT grade
       """
   ```

3. **Update `_run_single_config` in `src/experiment_runner.py`** (lines 124-135):
   Replace direct ID comparison with `compute_overlap_relevance()` call. Pass `retrieved_chunks` (which carry `metadata.start_char/end_char`) instead of just IDs.

4. **Update ground truth generation script** to record `document_id`, `start_char`, `end_char` for each referenced chunk (available from `chunk.metadata`).

5. **Update ground truth curation instructions** — developer assigns grades to document sections, not chunk IDs.

**Validation:** After implementing, run a smoke test: chunk one document with FixedChunker AND RecursiveChunker, compute metrics for a known query against both — both should produce non-zero metrics if chunks overlap with ground truth sections.

**Files:** `src/schemas.py`, `src/evaluation/metrics.py`, `src/experiment_runner.py`, `scripts/generate_ground_truth.py`, tests

---

## Path & Signature Verification (FIX 2-5)

All paths and signatures verified against actual repo — plan is already correct:
- **FIX 2 (config dir):** Confirmed `experiments/configs/` (not `src/configs/`). Day 3 handover was wrong.
- **FIX 3 (ground truth path):** Confirmed `data/ground_truth.json` (not `data/ground_truth/`). Day 3 handover was wrong.
- **FIX 4 (results output):** Confirmed `results/experiments/summary.json`. Runner defaults match plan.
- **FIX 5 (generate signature):** Confirmed `generate_ground_truth_candidates(chunks, n=30, model="gpt-4o")` — takes `model` string, not `llm` object. Day 3 handover was wrong.

No changes needed — paths and signatures in this plan are correct.

---

## Step 0b: PDF Extraction with Vision LLM Image Descriptions — DEVELOPER GATE

Before generating ground truth, extract all 4 PDFs with image/figure descriptions and save to disk for manual inspection.

**Enhanced extraction pipeline (`src/extraction.py`):**
1. `extract_pdf(path, describe_images=True)` extracts text per page as before, PLUS:
   - Renders each page as PNG via `page.get_pixmap(dpi=150)`
   - Sends to GPT-4o-mini vision via litellm with a prompt to describe figures, tables, diagrams
   - Appends descriptions to page text as `[Visual Content — Page N]\n{description}`
   - Pages with no visual elements (model responds `NO_VISUAL_ELEMENTS`) get no appended text
   - Errors on individual pages are caught and skipped (logged, not fatal)
2. `save_document(doc, path)` / `load_document(path)` — JSON serialization for disk caching
3. `extract_all_pdfs(pdf_dir, describe_images=True, cache_dir="data/extracted")` — cache layer:
   - If `data/extracted/{stem}.json` exists → load from cache (zero API cost)
   - Otherwise → extract with vision → save JSON + .txt to cache → return
   - `--force` flag bypasses cache for re-extraction

**Files committed to repo:**
- `data/pdfs/*.pdf` — source PDFs (un-gitignored, committed for reproducibility)
- `data/extracted/{stem}.json` — canonical cached Documents (with image descriptions)
- `data/extracted/{stem}.txt` — human-readable text (for manual inspection)

**Inspection script (`scripts/inspect_extraction.py`):**
- Prints per-document stats: page count, char count, empty/short page warnings
- Usage: `python scripts/inspect_extraction.py --describe-images` (first run)
- Subsequent runs load from cache automatically

**Cost:** ~$0.01 for 4 papers (61 pages × GPT-4o-mini vision)

**Developer: review `data/extracted/*.txt` files for:**
1. Image descriptions are reasonable (check pages with known figures)
2. No garbled text or encoding issues
3. Total char count increased from ~216K (text-only) to ~240K (with descriptions)

**Wait for developer confirmation before proceeding to Step 1.**

---

## Step 1: Ground Truth Curation

**Create `scripts/generate_ground_truth.py`:**
1. `extract_all_pdfs("data/pdfs/")` → 4 Documents
2. Chunk with RecursiveChunker(512, 50) — good general-purpose baseline
3. `generate_ground_truth_candidates(chunks, n=30, model="gpt-4o")` → 30 QA pairs
4. Write candidates to `data/ground_truth_candidates.json`. Format each candidate as:
   ```json
   {
     "query_id": "q01",
     "question": "...",
     "source_pdf": "attention.pdf",
     "suggested_answer": "...",
     "relevant_sections": [
       {
         "chunk_id": "abc123",
         "document_id": "doc_hash",
         "start_char": 4500,
         "end_char": 5012,
         "text_preview": "first 100 chars of the chunk...",
         "suggested_grade": 3
       }
     ]
   }
   ```
   Include `text_preview` (first 100 chars) so the developer can identify sections
   without cross-referencing the full extracted text. Include `document_id`,
   `start_char`, `end_char` because the cross-chunker matching (FIX 1) requires
   these fields in the final ground_truth.json.

5. **STOP — Developer curates ground truth manually.** Do NOT auto-accept LLM output.
   Print this message and wait:

   ```
   === GROUND TRUTH CURATION — DEVELOPER REQUIRED ===
   30 LLM-generated candidates saved to data/ground_truth_candidates.json

   Open the file and perform these steps:
   a) DELETE questions answerable from general knowledge (not PDF-specific)
   b) DELETE questions where the answer is trivially in one obvious chunk
      (too easy — won't discriminate between retrieval configs)
   c) DELETE duplicate or near-duplicate questions
   d) VERIFY remaining queries span all 4 papers (minimum 2 per paper)
   e) VERIFY difficulty mix: ~5 easy (1 gold chunk), ~5 medium (2 chunks),
      ~5 hard (synthesis across sections)
   f) For each kept query, review the relevant_sections and assign relevance grades:
      3 = directly answers | 2 = same section | 1 = same document | 0 = irrelevant
      These grades apply to DOCUMENT SECTIONS (char offset ranges), not chunk IDs.
      The text_preview field shows what each section contains.
      You may adjust start_char/end_char boundaries if the LLM picked too narrow
      or too wide a span. You may also add additional relevant sections the LLM missed.
   g) Target: 15 queries minimum

   Save curated result as data/ground_truth.json matching GroundTruthSet schema.
   Validate: python -c "from src.evaluation.ground_truth import load_ground_truth; \
     gt = load_ground_truth('data/ground_truth.json'); \
     print(f'{len(gt.queries)} queries loaded, valid')"

   Reply 'continue' when ground_truth.json is validated.
   ```

6. Wait for developer confirmation that `load_ground_truth()` succeeds before proceeding to Step 2.

**Key:** Deterministic chunk IDs (Step 0) are still needed so that reproducibility checks
produce consistent results within the SAME config. Cross-chunker matching (FIX 1) handles
the case where DIFFERENT chunkers produce different IDs for overlapping content.

---

## Step 2: Add Missing Configs (Reranking + Alpha Sweep)

**Reranking configs (for Q3 — reranking before/after):**
- `36_recursive_minilm_dense_rerank.yaml` — use_reranking: true, reranker_type: cross_encoder
- `37_recursive_mpnet_dense_rerank.yaml`
- `38_recursive_minilm_hybrid_rerank.yaml`
- `39_recursive_mpnet_hybrid_rerank.yaml`

**Alpha sweep configs (for hybrid alpha chart):**
- `40_recursive_minilm_hybrid_a03.yaml` — hybrid_alpha: 0.3
- `41_recursive_minilm_hybrid_a05.yaml` — hybrid_alpha: 0.5
- `42_recursive_minilm_hybrid_a09.yaml` — hybrid_alpha: 0.9
- (alpha=0.7 already exists as config 12)

**Note:** Alpha sweep uses recursive + minilm as a fixed baseline. If after the grid run a
different chunker+embedder combo performs best, consider re-running the alpha sweep with that
combo. Document this assumption in the comparison report methodology section.

**Fix experiment runner:** `_run_single_config` does NOT apply reranking even when `config.use_reranking=True`. After retrieval (line ~112), add:
```python
if config.use_reranking and config.reranker_type:
    reranker = create_reranker(config.reranker_type)
    retrieval_results = reranker.rerank(query, retrieval_results, top_k=config.top_k)
```
Import `create_reranker` from `src.factories`.

**Files:** `experiments/configs/36-42_*.yaml`, `src/experiment_runner.py`

**PRD 6b gap:** PRD specifies "best 3 configs × 2 rerankers" for extended experiments.
Day 4 covers cross_encoder only (4 configs). If COHERE_API_KEY is set, add 4 matching
Cohere reranking configs (43-46) for a proper reranker comparison. If unavailable,
document in comparison report that only cross-encoder reranking was tested.

---

## Step 3: Full Grid Run

Run `run_experiment_grid()` across all ~42 configs with:
- `documents = extract_all_pdfs("data/pdfs/")`
- `run_judge=True` (gpt-4o for 5-axis scoring)
- Results saved to `results/experiments/summary.json`

**Entry point: `scripts/evaluate.py`** — Click CLI with flags:
- `--configs`, `--ground-truth`, `--output`, `--pdfs`
- `--no-judge` (skip LLM judge)
- `--reproducibility-check` (re-run best config twice)

**Cost tracking:** Verify that `_run_single_config` populates `performance.cost_estimate_usd`.
For OpenAI embedder configs, estimate based on token count × pricing ($0.02/1M tokens for
text-embedding-3-small). For local embedders, cost = 0. For LLM generation, estimate based on
gpt-4o-mini pricing. If cost tracking is missing, add it before running the full grid.

**Time Estimation Checkpoint:**
After the first 4 configs complete, log wall-clock time and print:
`"4 configs completed in {X}m. Estimated total for {N} configs: ~{Y}m."`
If estimated total exceeds 3 hours, print a warning and suggest:
`"WARNING: Grid will take ~{Y}m. Consider running with --no-judge first`
`(retrieval metrics only), then re-running judge on top-5 configs only."`

---

## Step 4: Visualization Module (`src/visualization.py`)

Replace stub with 10 chart functions + `generate_all_charts()` orchestrator. All save PNGs to `results/charts/`.

| # | Chart | Function | Answers |
|---|-------|----------|---------|
| 1 | Config x Metric heatmap | `plot_config_metric_heatmap` | Overview |
| 2 | Chunking comparison | `plot_chunking_comparison` | Q1 |
| 3 | Embedding comparison | `plot_embedding_comparison` | Q4 |
| 4 | Retriever comparison | `plot_retriever_comparison` | Q2 |
| 5 | Hybrid alpha sweep | `plot_hybrid_alpha_sweep` | — |
| 6 | Reranking before/after | `plot_reranking_comparison` | Q3 |
| 7 | NDCG@5 distribution | `plot_ndcg_distribution` | — |
| 8 | Judge 5-axis radar | `plot_judge_radar` | — |
| 9 | Latency vs quality scatter | `plot_latency_vs_quality` | — |
| 10 | Per-query difficulty | `plot_query_difficulty` | — |
| 11 | Local vs API embedding | `plot_local_vs_api` | Q5 — **DEFERRED to Day 5 (Ollama)** |

**Helpers needed:**
- `_config_label(config) -> str` — short label like `recursive_minilm_dense`
- `_results_to_dataframe(results) -> pd.DataFrame` — flatten for seaborn/matplotlib
- Use `matplotlib` + `seaborn`, style: `seaborn-v0_8-whitegrid`, figsize (12,8), dpi 150
- `generate_all_charts()` produces 10 charts on Day 4. Chart 11 is added in Day 5 when Ollama experiment data exists.

---

## Step 5: Comparison Report (`src/reporting.py`)

`generate_comparison_report(results, output_path="results/comparison_report.md") -> Path`

Markdown report with sections:
- **Summary** — total configs, best config, target met/missed
- **Q1** — chunking strategy table (control for embedder+retriever)
- **Q2** — dense vs BM25 vs hybrid table
- **Q3** — reranking before/after delta table
- **Q4** — embedding model comparison with latency + cost columns
- **Best Configuration** — full YAML dump of the winning config
- **Methodology**
- **Iteration Log Table** — rendered from `results/iteration_log.json`, showing each single-parameter change with before/after metrics and delta (PRD 7g)
- **Final Config Traceability** — table mapping every component choice in the best config to the specific experiment pair that justified it (PRD 7g)
- **Judge Target Check** — does best config achieve avg > 4.0 across all 5 axes? (PRD 2b target). If not, flag which axes fell short and by how much.
- **Self-Evaluation Answers** — draft answers to PRD Section 8c questions 1-5, each referencing specific experiment IDs. (Q6 is Day 5 — Ollama.)

---

## Step 6: Iteration Log (`src/iteration_log.py`)

**Schema:**
```python
class IterationEntry(BaseModel):
    iteration_id: int
    parameter_changed: str
    old_value: str
    new_value: str
    reason: str
    experiment_id_before: str
    experiment_id_after: str
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    delta: dict[str, float]
```

`build_iteration_log(results)` — finds config pairs differing by exactly one parameter, records deltas.
Saves to `results/iteration_log.json`.

---

## Step 7: Reproducibility Check

Add to `src/experiment_runner.py`:
```python
def run_reproducibility_check(results, documents, ground_truth_path, threshold=0.05) -> dict
```
1. Find best config by NDCG@5
2. Re-run `_run_single_config()` (no judge — avoids LLM variance)
3. Compare retrieval metrics: `|run1 - run2| / max(run1, 0.001) < threshold`
4. Return pass/fail per metric

Triggered by `--reproducibility-check` flag in CLI.

---

## Step 8: Judge Calibration (`scripts/judge_calibration.py`)

1. Pick 5 diverse query-answer pairs from grid results (different configs, different papers)
2. Write them to `results/judge_calibration_input.json` with: query, answer, context chunks, LLM judge scores
3. **STOP and print:**

   ```
   === JUDGE CALIBRATION — DEVELOPER REQUIRED ===
   5 query-answer pairs saved to results/judge_calibration_input.json

   For each pair, read the answer + context chunks and score on 5 axes (1-5):
   - Relevance: 1=off-topic → 5=directly answers
   - Accuracy: 1=major errors → 5=every claim verifiable
   - Completeness: 1=fragment → 5=comprehensive
   - Conciseness: 1=verbose → 5=focused
   - Citation Quality: 1=no citations → 5=every claim cited

   Add your scores as "human_scores" to each entry in the JSON file.
   Reply 'continue' when done.
   ```

4. After developer continues, load the file, compare human vs LLM per-axis, compute deltas
5. Save final output to `results/judge_calibration.json`

---

## Step 9: Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_visualization.py` | Each chart function creates PNG, generate_all returns 10 paths, edge cases |
| `tests/test_reporting.py` | Report creates MD, contains Q1-Q4, best config ID |
| `tests/test_iteration_log.py` | Single-param diff detection, delta math, empty input |
| `tests/test_reproducibility.py` | Pass/fail thresholds, best config selection (mock _run_single_config) |
| `tests/test_chunkers.py` (additions) | Deterministic IDs: same input → same output |
| `tests/test_integration.py` | Full pipeline: PDF → chunk → embed → index → retrieve → generate → cited answer. Uses 1 test PDF, 1 config, 1 query. Verifies QAResponse has non-empty answer, valid citations, latency < 30s. |

---

## Execution Order
1. Deterministic chunk IDs (blocker for everything)
2. Verify + fix paths: config dir, ground truth path, results output path (FIX 2-4)
3. Resolve ground truth cross-chunker matching strategy (FIX 1 — CRITICAL)
4. PDF extraction quality check — WAIT for developer approval
5. Ground truth generation script (correct function signature per FIX 5) — WAIT for developer curation
6. Reranking fix in experiment_runner + new YAML configs (correct path)
7. Verify cost tracking exists in _run_single_config (FIX 11)
8. `scripts/evaluate.py` CLI entry point
9. Run full grid (42+ configs) — check time estimate after first 4
10. `src/visualization.py` — 10 charts (chart 11 deferred to Day 5)
11. `src/reporting.py` — comparison report + judge target check + self-eval answers
12. `src/iteration_log.py` — iteration log
13. Reproducibility check
14. Judge calibration — WAIT for developer scoring (FIX 7)
15. Tests for all new code including integration test (FIX 6), maintain 95%+
16. Update CLAUDE.md, commit docs/plans/day4-plan.md

---

## Verification
- [ ] All file paths verified against actual repo (configs, ground truth, results output)
- [ ] Ground truth cross-chunker matching strategy implemented and tested
- [ ] PDF extraction quality inspected and approved before ground truth generation
- [ ] `load_ground_truth()` succeeds (15+ queries, valid chunk IDs, spans all 4 papers, difficulty mix)
- [ ] `run_experiment_grid()` completes for all configs, results JSON exists at correct path
- [ ] Non-recursive chunker configs produce non-zero retrieval metrics (FIX 1 validation)
- [ ] 10 PNGs in `results/charts/` (chart 11 deferred to Day 5)
- [ ] `results/comparison_report.md` answers Q1-Q4 with data tables
- [ ] Best config judge scores checked against > 4.0 avg target (PRD 2b)
- [ ] `results/iteration_log.json` traces config decisions
- [ ] Reproducibility check passes (<5% variance)
- [ ] `results/judge_calibration.json` compares human vs LLM on 5 answers
- [ ] Every ExperimentResult includes performance fields: ingestion_time_seconds, avg_query_latency_ms, index_size_bytes, peak_memory_mb, embedding_source, cost_estimate_usd (OpenAI configs > $0, BM25 = $0)
- [ ] Integration test passes: PDF → chunk → embed → retrieve → generate → cited answer
- [ ] `data/debug/` added to .gitignore
- [ ] `pytest` passes, coverage ≥95%
