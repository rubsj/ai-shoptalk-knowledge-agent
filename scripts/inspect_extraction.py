"""PDF extraction quality check — DEVELOPER GATE before ground truth generation.

Extracts all PDFs in data/pdfs/ and prints a quality report:
  - Per-document: page count, total chars, empty page count, title/author
  - Per-page: char count and flag for suspiciously short pages (<100 chars)
  - First 500 chars of each document (visual sanity check)

Usage:
    python scripts/inspect_extraction.py
    python scripts/inspect_extraction.py --pdf-dir data/pdfs --preview-chars 300
    python scripts/inspect_extraction.py --save-dir data/extracted

Developer should verify:
  - All 4 PDFs extracted with no failures
  - No pages with 0 chars (image-only pages that PyMuPDF couldn't OCR)
  - First 500 chars look like running text, not garbled encoding artifacts
  - Total char counts are plausible (~5K–80K chars per academic paper)

Exit code 0 = all checks pass. Exit code 1 = at least one failure/warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path so `src` resolves without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction import extract_all_pdfs  # noqa: E402

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# Pages shorter than this are flagged as suspicious
_MIN_PAGE_CHARS = 100

# Flag documents shorter than this total (may indicate extraction failure)
_MIN_DOC_CHARS = 5_000


def _print_document_report(doc, preview_chars: int) -> list[str]:
    """Print stats for one document. Returns list of warning strings."""
    warnings: list[str] = []

    total_chars = sum(p.char_count for p in doc.pages)
    empty_pages = [p.page_number for p in doc.pages if p.char_count == 0]
    short_pages = [p.page_number for p in doc.pages if 0 < p.char_count < _MIN_PAGE_CHARS]

    print(f"\n{_SEPARATOR}")
    print(f"  FILE   : {doc.metadata.source}")
    print(f"  ID     : {doc.id}")
    print(f"  TITLE  : {doc.metadata.title or '(none in PDF metadata)'}")
    print(f"  AUTHOR : {doc.metadata.author or '(none in PDF metadata)'}")
    print(f"  PAGES  : {doc.metadata.page_count}")
    print(f"  CHARS  : {total_chars:,}")
    print(_SUBSEP)

    if total_chars < _MIN_DOC_CHARS:
        msg = f"WARNING: only {total_chars:,} chars total — possible extraction failure"
        print(f"  ⚠  {msg}")
        warnings.append(msg)

    if empty_pages:
        msg = f"WARNING: {len(empty_pages)} empty page(s) (image-only?): {empty_pages[:10]}"
        print(f"  ⚠  {msg}")
        warnings.append(msg)

    if short_pages:
        msg = f"WARNING: {len(short_pages)} short page(s) (<{_MIN_PAGE_CHARS} chars): {short_pages[:10]}"
        print(f"  ⚠  {msg}")
        warnings.append(msg)

    if not warnings:
        print("  ✓  No extraction warnings")

    # Per-page breakdown
    print(f"\n  {'PAGE':>4}  {'CHARS':>7}  STATUS")
    print(f"  {'----':>4}  {'-------':>7}  ------")
    for p in doc.pages:
        status = ""
        if p.char_count == 0:
            status = "EMPTY ⚠"
        elif p.char_count < _MIN_PAGE_CHARS:
            status = f"SHORT ({p.char_count} chars) ⚠"
        print(f"  {p.page_number:>4}  {p.char_count:>7,}  {status}")

    # Content preview
    print(f"\n  PREVIEW (first {preview_chars} chars of document.content):")
    print(_SUBSEP)
    preview = doc.content[:preview_chars].replace("\n", "↵ ")
    print(f"  {preview}")
    print(_SUBSEP)

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF extraction quality check")
    parser.add_argument(
        "--pdf-dir",
        default="data/pdfs",
        help="Directory containing PDFs to inspect (default: data/pdfs)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of chars to preview per document (default: 500)",
    )
    parser.add_argument(
        "--save-dir",
        default="data/extracted",
        help="Directory to save extracted text files (default: data/extracted)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    print(_SEPARATOR)
    print("  PDF EXTRACTION QUALITY REPORT")
    print(f"  Directory: {pdf_dir.resolve()}")
    print(_SEPARATOR)

    print("\nExtracting PDFs...", end=" ", flush=True)
    try:
        documents = extract_all_pdfs(pdf_dir)
    except Exception as exc:
        print(f"\nERROR during extraction: {exc}", file=sys.stderr)
        return 1
    print(f"done ({len(documents)} documents)")

    # Save extracted text to disk so developer can physically inspect
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        stem = Path(doc.metadata.source).stem
        out_file = save_dir / f"{stem}.txt"
        out_file.write_text(doc.content, encoding="utf-8")
    print(f"Saved extracted text to {save_dir.resolve()}/")

    all_warnings: list[str] = []
    for doc in documents:
        warnings = _print_document_report(doc, args.preview_chars)
        all_warnings.extend(warnings)

    # Summary
    print(f"\n{_SEPARATOR}")
    print("  SUMMARY")
    print(_SEPARATOR)
    print(f"  Documents extracted : {len(documents)}")
    total_pages = sum(d.metadata.page_count for d in documents)
    total_chars = sum(len(d.content) for d in documents)
    print(f"  Total pages        : {total_pages}")
    print(f"  Total chars        : {total_chars:,}")

    if all_warnings:
        print(f"\n  ⚠  {len(all_warnings)} warning(s) — review before generating ground truth:")
        for w in all_warnings:
            print(f"     • {w}")
        print()
        return 1

    print("\n  ✓  All checks passed — safe to run scripts/generate_ground_truth.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
