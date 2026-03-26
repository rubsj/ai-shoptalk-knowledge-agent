"""Shared utilities for all chunking strategies.

Extracted here so each chunker doesn't reimplement page-number lookup or chunk ID generation.
"""

from __future__ import annotations

import hashlib

from src.schemas import Document


def make_chunk_id(document_id: str, start_char: int, end_char: int) -> str:
    """Return a deterministic 16-char hex ID for a chunk position.

    Same document + same char offsets → same ID across runs.
    WHY: ground truth stores chunk IDs; non-deterministic IDs (uuid4) would
    break metric computation every re-run.
    """
    key = f"{document_id}:{start_char}:{end_char}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def find_page_number(document: Document, char_offset: int) -> int:
    """Map a character offset in Document.content to its originating page number.

    Document.content is built by joining page texts with "\\n\\n" separators.
    This function reconstructs the per-page boundaries and returns the page
    whose text contains the given offset.

    Args:
        document: The source Document.
        char_offset: Character position in document.content (0-indexed).

    Returns:
        0-indexed page number. Returns the last page if offset is beyond the
        end of content (defensive — should not happen in normal use).
    """
    cursor = 0
    for page in document.pages:
        # page.text length + 2 for the "\n\n" separator added between pages
        # WHY: Document.content = "\n\n".join(p.text for p in pages)
        # so each page occupies exactly len(page.text) chars, then 2 more for "\n\n"
        # (except the last page which has no trailing separator)
        page_end = cursor + len(page.text)
        if char_offset <= page_end:
            return page.page_number
        # Move past this page's text and the "\n\n" separator
        cursor = page_end + 2  # +2 for "\n\n"

    # char_offset is beyond all page content — return last page (defensive)
    return document.pages[-1].page_number
