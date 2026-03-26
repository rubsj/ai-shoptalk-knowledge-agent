# ADR-006: PDF Extraction with Vision LLM Image Descriptions

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Data Extraction Pipeline
**Status:** Accepted
**Date:** 2026-03-26

---

## Context

The RAG system ingests 4 academic NLP papers (Attention, BERT, RAG, Sentence-BERT) as its knowledge base. PyMuPDF's `page.get_text()` extracts text but silently drops all visual content — figures, tables, architecture diagrams, and result charts. For papers like "Attention Is All You Need", the Transformer architecture diagram (Figure 1) is a core piece of content that a retrieval system should be able to surface when a user asks "how does the Transformer architecture work?"

Initial extraction produced 215,978 characters of text. Manual inspection of the extracted `.txt` files confirmed clean text but zero figure content — questions about visual elements would have no retrievable answer.

Three options were considered:

1. **Extract images as separate files** — dump PNGs alongside text. User can see them, but RAG can't retrieve or reason about them.
2. **Describe images with a vision LLM** — render pages as PNG, send to GPT-4o-mini vision, insert text descriptions into the document content. RAG can retrieve and reason about figure content.
3. **Keep text-only** — scope ground truth questions to text-answerable topics only.

## Decision

**Option 2: Vision LLM (GPT-4o-mini) describes figures, tables, and diagrams. Descriptions are interleaved at the correct vertical position on each page.**

### Sub-decisions

**D1: One vision call per page (not per image)**
Sending the full rendered page gives the LLM context for axis labels, legends, and captions that sit in text blocks near the figure. Individual cropped images lose this spatial context. Cost is negligible: 61 pages x GPT-4o-mini vision ≈ $0.01.

**D2: Positional interleaving via `get_text("dict")` block bounding boxes**
Initial implementation appended image descriptions at the end of each page's text. This broke the page's reading order — if Figure 1 appears between paragraphs 2 and 3, the description landed after paragraph 5. When chunked, the description could end up in a different chunk than the surrounding discussion of the figure.

Fix: PyMuPDF's `page.get_text("dict")` returns both text blocks (type=0) and image blocks (type=1) with bounding boxes. Sorting all blocks by y-coordinate and inserting the description at the first image block's position preserves reading order.

Key finding: `get_text("blocks")` (simpler tuple format) does NOT return image blocks for many PDFs — it silently omits XObject images. Only `get_text("dict")` reliably detects them. This was discovered empirically when the first interleaving attempt produced zero image descriptions.

**D3: `[Visual Content — Page N]` marker format**
Descriptions are wrapped in a `[Visual Content — Page N]` marker so they're identifiable in the extracted text, distinguishable from original paper content, and searchable. The marker also helps during ground truth curation — the developer can see which descriptions came from the vision LLM.

**D4: Disk cache with JSON serialization (extract once, reuse everywhere)**
Vision API calls cost money and take time (~5s per page). Extracted Documents are serialized as JSON to `data/extracted/{stem}.json`. Subsequent calls to `extract_all_pdfs()` load from cache with zero API cost. The cache is committed to the repo so CI and collaborators never need to re-extract.

Human-readable `.txt` copies are saved to `data/extracted/validation/` for manual inspection.

**D5: `describe_images=False` default (backward compatible)**
`extract_pdf()` defaults to text-only extraction. Vision calls require explicit opt-in (`describe_images=True`). This means:
- All existing tests pass unchanged (they use the text-only path)
- Fast local development doesn't hit the API
- The cached JSON files contain the enriched content for experiment runs

**D6: Commit PDFs and extracted content to repo**
Previously, `data/pdfs/*.pdf` was gitignored ("download locally"). Changed to committed because:
- Reproducibility: anyone cloning the repo gets identical source data
- Cache validity: the JSON cache is only valid for these specific PDFs
- Size is acceptable: 4.3MB of PDFs + 500K of JSON cache

## Consequences

**Positive:**
- RAG system can now retrieve and answer questions about figures, tables, and architecture diagrams
- Ground truth questions can reference visual content (e.g., "What components are shown in the Transformer architecture?")
- Extraction cost is one-time ($0.01), amortized to zero via disk cache
- Page reading order is preserved — chunkers produce coherent chunks that include figure descriptions in their natural context

**Negative:**
- Vision LLM descriptions are imperfect — may miss fine-grained table values or misread axis labels
- `get_text("dict")` text reconstruction (lines → spans) may differ slightly from `get_text()` output for pages with images, though cleaning normalizes most differences
- For pages with multiple figures, all descriptions are inserted at the position of the first image block (adequate for academic papers where figures are typically well-separated)

**Trade-offs accepted:**
- Single description block per page vs. per-figure descriptions — simpler, and most academic pages have 1-2 figures
- GPT-4o-mini vs. GPT-4o for vision — cheaper, sufficient quality for describing diagrams (not OCR-ing fine print)

## Technical Details

**Files modified:**
- `src/extraction.py` — `_describe_page_images()`, `_extract_page_content()`, `_extract_text_from_dict_block()`, `save_document()`, `load_document()`, updated `extract_all_pdfs()`
- `scripts/inspect_extraction.py` — `--describe-images`, `--force` flags
- `.gitignore` — un-ignored `data/pdfs/*.pdf` and `data/extracted/`

**Extraction stats:**
| Paper | Pages | Text-only chars | With image descriptions |
|-------|-------|----------------|------------------------|
| Attention Is All You Need | 15 | 39,447 | 42,142 |
| BERT | 16 | 63,577 | 66,192 |
| RAG | 19 | 69,062 | 72,893 |
| Sentence-BERT | 11 | 43,778 | 43,778 |
| **Total** | **61** | **215,978** | **225,119** |

Sentence-BERT has no XObject images detected by dict mode (its figures may be vector-drawn paths rather than embedded raster images).
