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

*Entries will be added as Day 4 work continues through ground truth generation, experiment grid execution, visualization, and analysis.*
