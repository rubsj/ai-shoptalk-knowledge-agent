"""PDF extraction with PyMuPDF (fitz) and text cleaning.

Pipeline:
    extract_pdf()  →  Document (all pages, cleaned text, metadata)
    extract_all_pdfs()  →  list[Document]

Per-page cleaning (in order):
    1. remove_headers_footers()  — drop running headers/page numbers
    2. clean_text()              — fix ligatures, rejoin hyphenated words, collapse whitespace

Optional: describe_images=True renders each page as PNG and sends to a vision
LLM (GPT-4o-mini) to describe figures, tables, and diagrams. Descriptions are
appended to the page text so downstream chunking/retrieval can surface them.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF — `pip install pymupdf` installs as `fitz`
import litellm

from src.schemas import Document, DocumentMetadata, PageInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled regexes — compiled once at module load, not per-call.
# Matters when batch-processing hundreds of pages.
# ---------------------------------------------------------------------------

# PDF fonts sometimes store fi/fl as single unicode ligature glyphs
_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# PDFs break "trans-\nformer" across lines; NLP tokenizers see two tokens
_HYPHENATION_RE = re.compile(r"(\w+)-\n(\w+)")

_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_EXCESSIVE_SPACES_RE = re.compile(r"[ \t]{2,}")

# Header/footer heuristics
_STANDALONE_DIGIT_RE = re.compile(r"^\s*\d+\s*$")               # "  3  "
_PAGE_N_RE = re.compile(r"^\s*[Pp]age\s+\d+", re.IGNORECASE)    # "Page 3", "page 3 of 10"
_N_OF_M_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE)  # "3 of 10"
_ARXIV_RE = re.compile(r"^\s*arXiv:", re.IGNORECASE)             # "arXiv:1706.03762v5"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Fix ligatures, rejoin hyphenated line-breaks, collapse whitespace.

    Returns text safe for NLP tokenization.
    """
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    # r"\1\2" joins captured groups — "re-\njoin" → "rejoin"
    text = _HYPHENATION_RE.sub(r"\1\2", text)

    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)
    text = _EXCESSIVE_SPACES_RE.sub(" ", text)

    return text.strip()


def _is_header_or_footer(line: str, page_number: int, total_pages: int) -> bool:
    """Check if a line matches common header/footer patterns.

    page_number/total_pages unused but kept for call-site symmetry with
    remove_headers_footers().
    """
    stripped = line.strip()
    if not stripped:
        return False

    if _STANDALONE_DIGIT_RE.match(stripped):
        return True
    if _PAGE_N_RE.match(stripped):
        return True
    if _N_OF_M_RE.match(stripped):
        return True
    if _ARXIV_RE.match(stripped):
        return True

    # Short ALL-CAPS line — journal/conference headers like "NEURAL INFORMATION PROCESSING SYSTEMS"
    # ≥3 chars avoids false-positives on single-letter section labels ("A")
    if 3 <= len(stripped) <= 60 and stripped.upper() == stripped and stripped.replace(" ", "").isalpha():
        return True

    return False


def remove_headers_footers(text: str, page_number: int, total_pages: int) -> str:
    """Strip running headers/footers by inspecting first 3 and last 3 lines.

    Skips pages with ≤6 lines — removing 6 of 6 lines would nuke the page.
    """
    lines = text.split("\n")

    if len(lines) <= 6:
        return text

    head_keep = []
    for i, line in enumerate(lines[:3]):
        if not _is_header_or_footer(line, page_number, total_pages):
            head_keep = lines[i:]
            break
    else:
        head_keep = lines[3:]

    foot_keep = []
    for i, line in enumerate(reversed(head_keep[-3:])):
        if not _is_header_or_footer(line, page_number, total_pages):
            cut = len(head_keep) - i
            foot_keep = head_keep[:cut]
            break
    else:
        foot_keep = head_keep[:-3] if len(head_keep) > 3 else head_keep

    return "\n".join(foot_keep)


_VISION_SYSTEM_PROMPT = """\
You are analyzing a page from an academic paper. Describe any figures, tables, \
charts, diagrams, or other visual elements on this page.

For each visual element:
1. Identify it (e.g., "Figure 3", "Table 2")
2. Describe what it shows in detail
3. Include any axis labels, legends, column headers, or data values visible

If this page contains only text and mathematical equations with no figures, \
tables, or diagrams, respond with exactly: NO_VISUAL_ELEMENTS"""

_NO_VISUALS_MARKER = "NO_VISUAL_ELEMENTS"


def _describe_page_images(
    page_png_bytes: bytes,
    page_number: int,
    vision_model: str = "gpt-4o-mini",
) -> str:
    """Send a rendered page image to a vision LLM and get figure descriptions.

    Returns description text, or empty string if no visual elements found.
    """
    b64_image = base64.b64encode(page_png_bytes).decode("utf-8")

    response = litellm.completion(
        model=vision_model,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": f"This is page {page_number + 1}. Describe any visual elements.",
                    },
                ],
            },
        ],
        temperature=0.0,
    )
    content: str = response.choices[0].message.content
    if _NO_VISUALS_MARKER in content:
        return ""
    return content


def extract_pdf(
    pdf_path: str | Path,
    describe_images: bool = False,
    vision_model: str = "gpt-4o-mini",
) -> Document:
    """Extract text and metadata from a single PDF.

    Args:
        pdf_path: Path to the PDF file.
        describe_images: If True, render each page as PNG and send to a vision
            LLM to describe figures/tables/diagrams. Descriptions are appended
            to the page text.
        vision_model: LiteLLM model name for vision calls (default: gpt-4o-mini).

    Raises FileNotFoundError if path is missing, ValueError if no text found.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))  # type: ignore[attr-defined]

    # grab metadata before close() — accessing after close raises RuntimeError
    raw_meta = doc.metadata or {}
    total_pages = len(doc)

    pages: list[PageInfo] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text()  # "" for image-only pages
        cleaned = clean_text(remove_headers_footers(raw_text, page_num, total_pages))

        # Optionally describe visual elements via vision LLM
        if describe_images:
            try:
                pixmap = page.get_pixmap(dpi=150)  # type: ignore[attr-defined]
                png_bytes = pixmap.tobytes("png")
                description = _describe_page_images(png_bytes, page_num, vision_model)
                if description:
                    cleaned = cleaned + f"\n\n[Visual Content — Page {page_num + 1}]\n{description}"
                    logger.debug("Page %d: added image description (%d chars)", page_num, len(description))
            except Exception:
                logger.exception("Page %d: failed to describe images, skipping", page_num)

        pages.append(
            PageInfo(
                page_number=page_num,
                text=cleaned,
                char_count=len(cleaned),
            )
        )

    doc.close()

    full_content = "\n\n".join(p.text for p in pages)
    if not full_content.strip():
        raise ValueError(f"No extractable text found in PDF: {path}")

    metadata = DocumentMetadata(
        source=str(path),
        title=raw_meta.get("title", "") or "",
        author=raw_meta.get("author", "") or "",
        page_count=total_pages,
    )

    return Document(content=full_content, metadata=metadata, pages=pages)


# ---------------------------------------------------------------------------
# Disk cache — extract once, reuse everywhere
# ---------------------------------------------------------------------------


def save_document(document: Document, output_path: str | Path) -> Path:
    """Save a Document as JSON to disk. Returns the written path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(document.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    return out


def load_document(path: str | Path) -> Document | None:
    """Load a Document from a JSON file. Returns None if file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Document.model_validate(data)


def extract_all_pdfs(
    pdf_dir: str | Path,
    describe_images: bool = False,
    cache_dir: str | Path = "data/extracted",
    force: bool = False,
) -> list[Document]:
    """Extract all PDFs in a directory, using disk cache when available.

    Cache behaviour:
        - If cache_dir/{stem}.json exists and force=False → load from cache.
        - Otherwise → extract (optionally with image descriptions) → save to cache.
        - Also saves a human-readable .txt to cache_dir/validation/ for inspection.

    Args:
        pdf_dir: Directory containing PDF files.
        describe_images: If True, describe figures/tables via vision LLM on cache miss.
        cache_dir: Directory for cached extraction results.
        force: If True, re-extract even if cache exists.
    """
    pdf_dir = Path(pdf_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        stem = pdf_path.stem
        json_cache = cache_path / f"{stem}.json"

        # Try loading from cache
        if not force and json_cache.exists():
            cached_doc = load_document(json_cache)
            if cached_doc is not None:
                logger.info("Loaded from cache: %s", json_cache.name)
                documents.append(cached_doc)
                continue

        # Cache miss — extract
        try:
            doc = extract_pdf(pdf_path, describe_images=describe_images)
            # Save JSON (canonical) and .txt (human-readable validation copy)
            save_document(doc, json_cache)
            validation_dir = cache_path / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            (validation_dir / f"{stem}.txt").write_text(doc.content, encoding="utf-8")
            logger.info("Extracted and cached %s: %d pages", pdf_path.name, doc.metadata.page_count)
            documents.append(doc)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping %s: %s", pdf_path.name, e)

    return documents