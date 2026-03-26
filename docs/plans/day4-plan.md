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

## Step 0b: PDF Extraction Quality Check — DEVELOPER GATE

Before generating ground truth, extract all 4 PDFs and write raw text to disk for manual inspection.

**Create `scripts/inspect_extraction.py`:**
1. For each PDF in `data/pdfs/`, call `extract_pdf(path)` → `Document`
2. Write `doc.content` to `data/debug/extracted_{stem}.txt` (create `data/debug/` if missing)
3. Print summary table: filename, page_count, total_chars, total_words
4. Print first 500 chars of page 1 and first 500 chars of the last page for each PDF

**Run the script, then STOP and print this message:**

```
=== PDF EXTRACTION QUALITY CHECK ===
Raw extracted text saved to data/debug/extracted_*.txt

Developer: open each file and check for:
1. COLUMN INTERLEAVING — sentences cut off mid-word then resume with
   unrelated text (two-column PDF merging). Look at paragraph boundaries.
2. HEADER/FOOTER CONTAMINATION — does paper title or "Page N" repeat
   every ~3000 chars? If yes, clean_pdf_text() needs a strip pattern.
3. LIGATURE ARTIFACTS — search for ﬁ, ﬂ, ﬃ characters. Should be
   normalized to fi, fl, ffi. If present, clean_pdf_text() is incomplete.
4. TABLE GARBAGE — columns of numbers with broken whitespace. Tables
   won't chunk well; acceptable but note if it dominates content.
5. REFERENCE SECTION NOISE — bibliography at end of each paper.
   Acceptable, but check it doesn't exceed ~20% of total text per doc.
6. ENCODING ISSUES — \x sequences, replacement chars (�), or mojibake.

Expected: these are single-column arXiv papers, extraction should be
clean. If issues found, fix clean_pdf_text() before proceeding.

Reply 'continue' when inspection is complete.
```

**Wait for developer confirmation before proceeding to Step 1.**

---

## Step 1: Ground Truth Curation

**Create `scripts/generate_ground_truth.py`:**
1. `extract_all_pdfs("data/pdfs/")` → 4 Documents
2. Chunk with RecursiveChunker(512, 50) — good general-purpose baseline
3. `generate_ground_truth_candidates(chunks, n=30, model="gpt-4o")` → 30 QA pairs
4. Write candidates to `data/ground_truth_candidates.json`. Format each candidate as:
   `{query_id, question, source_chunk_ids, suggested_answer, source_pdf}`.

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
   f) For each kept query, assign relevance grades to referenced chunks:
      3 = directly answers | 2 = same section | 1 = same document | 0 = irrelevant
   g) Target: 15 queries minimum

   Save curated result as data/ground_truth.json matching GroundTruthSet schema.
   Validate: python -c "from src.evaluation.ground_truth import load_ground_truth; \
     gt = load_ground_truth('data/ground_truth.json'); \
     print(f'{len(gt.queries)} queries loaded, valid')"

   Reply 'continue' when ground_truth.json is validated.
   ```

6. Wait for developer confirmation that `load_ground_truth()` succeeds before proceeding to Step 2.

**Key:** Chunk IDs must be deterministic FIRST (Step 0), so the IDs in ground_truth.json match future experiment runs using the same chunking config.

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

**Helpers needed:**
- `_config_label(config) -> str` — short label like `recursive_minilm_dense`
- `_results_to_dataframe(results) -> pd.DataFrame` — flatten for seaborn/matplotlib
- Use `matplotlib` + `seaborn`, style: `seaborn-v0_8-whitegrid`, figsize (12,8), dpi 150

---

## Step 5: Comparison Report (`src/reporting.py`)

`generate_comparison_report(results, output_path="results/comparison_report.md") -> Path`

Markdown report with sections:
- **Summary** — total configs, best config, target met/missed
- **Q1** — chunking strategy table (control for embedder+retriever)
- **Q2** — dense vs BM25 vs hybrid table
- **Q3** — reranking before/after delta table
- **Q4** — embedding model comparison with latency + cost columns
- **Best Configuration** — full YAML dump + traceability table
- **Methodology**
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

1. Pick 5 diverse query-answer pairs from grid results
2. Print each with context chunks for manual scoring
3. Developer enters 5-axis scores (1-5) interactively
4. Compare human vs LLM judge, compute per-axis delta
5. Save to `results/judge_calibration.json`

---

## Step 9: Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_visualization.py` | Each chart function creates PNG, generate_all returns 10 paths, edge cases |
| `tests/test_reporting.py` | Report creates MD, contains Q1-Q4, best config ID |
| `tests/test_iteration_log.py` | Single-param diff detection, delta math, empty input |
| `tests/test_reproducibility.py` | Pass/fail thresholds, best config selection (mock _run_single_config) |
| `tests/test_chunkers.py` (additions) | Deterministic IDs: same input → same output |

---

## Execution Order
1. Deterministic chunk IDs (blocker for everything)
2. PDF extraction quality check — WAIT for developer approval
3. Ground truth generation script — WAIT for developer curation
4. Reranking fix in experiment_runner + new YAML configs
5. `scripts/evaluate.py` CLI entry point
6. Run full grid (42+ configs) — check time estimate after first 4
7. `src/visualization.py` — all 10 charts
8. `src/reporting.py` — comparison report + judge target check + self-eval answers
9. `src/iteration_log.py` — iteration log
10. Reproducibility check
11. Judge calibration
12. Tests for all new code (maintain 95%+)
13. Update CLAUDE.md, commit `docs/plans/day4-plan.md`

---

## Verification
- [ ] `load_ground_truth("data/ground_truth.json")` succeeds (15 queries, valid chunk IDs)
- [ ] `run_experiment_grid()` completes for all configs, `results/experiments/summary.json` exists
- [ ] 10+ PNGs in `results/charts/`
- [ ] `results/comparison_report.md` answers Q1-Q4 with data tables
- [ ] `results/iteration_log.json` traces config decisions
- [ ] Reproducibility check passes (<5% variance)
- [ ] `results/judge_calibration.json` compares 5 answers
- [ ] `pytest` passes, coverage ≥95%
- [ ] Every ExperimentResult JSON includes performance fields: ingestion_time_seconds, avg_query_latency_ms, index_size_bytes, peak_memory_mb, embedding_source
- [ ] Best config judge scores checked against > 4.0 avg target (PRD 2b)
- [ ] Ground truth queries span all 4 papers with difficulty mix (easy/medium/hard)
- [ ] PDF extraction quality inspected and approved before ground truth generation
