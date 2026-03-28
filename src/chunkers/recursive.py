"""Recursive character text splitter — tries separators in order of preference.

Splits on ["\n\n", "\n", ". ", " ", ""] in sequence, falling back to the next
separator when a chunk exceeds chunk_size. Preserves paragraph structure when
possible. Most practical general-purpose strategy.

Config params: chunk_size (default 512), chunk_overlap (default 50).
"""

from __future__ import annotations

from src.chunkers._utils import find_page_number, make_chunk_id
from src.interfaces import BaseChunker
from src.schemas import Chunk, ChunkMetadata, Document


class RecursiveChunker(BaseChunker):
    """Recursive text splitter that respects natural language boundaries.

    Algorithm:
        1. Try to split on the largest separator ("\n\n" = paragraph boundary).
        2. If any resulting segment exceeds chunk_size, recurse with the next separator.
        3. Base case: empty separator "" splits at the character level (chunk_size windows).
        4. After splitting, add overlap by prepending the last chunk_overlap chars of
           the previous chunk to the current chunk.

    Java parallel: like LangChain's RecursiveCharacterTextSplitter but implemented
    from first principles without LangChain. See ADR-002.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters from previous chunk to prepend to current chunk.
            separators: Ordered list of split separators (largest semantic unit first).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # WHY: paragraph → sentence → word → character fallback chain
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document using recursive separator strategy.

        Args:
            document: Fully extracted Document.

        Returns:
            List of Chunks with metadata.
        """
        raw_chunks = self._split_text(document.content, self.separators)

        # Track raw chunk positions BEFORE overlap merge.
        # WHY: _merge_with_overlap prepends overlap text, so the merged chunk
        # is NOT a verbatim substring of document.content — find() would fail.
        # Raw chunks ARE substrings, so we locate them first.
        raw_positions: list[int] = []
        search_start = 0
        for text in raw_chunks:
            pos = document.content.find(text, search_start)
            if pos == -1:
                pos = document.content.find(text)
            if pos == -1:
                pos = search_start
            raw_positions.append(pos)
            search_start = pos + 1

        merged = self._merge_with_overlap(raw_chunks)

        chunks: list[Chunk] = []
        chunk_index = 0
        for i, text in enumerate(merged):
            if not text.strip():
                continue
            # For chunk 0: starts at raw position.
            # For chunk i>0: overlap prepends chunk_overlap chars, so the chunk
            # covers from (raw_pos - overlap) through (raw_pos + raw_len).
            raw_start = raw_positions[i]
            raw_end = raw_start + len(raw_chunks[i])
            if i == 0 or self.chunk_overlap == 0:
                start = raw_start
            else:
                start = max(0, raw_start - self.chunk_overlap)
            end = raw_end
            page_number = find_page_number(document, start)
            chunks.append(
                Chunk(
                    id=make_chunk_id(document.id, start, end),
                    content=text,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        source=document.metadata.source,
                        page_number=page_number,
                        start_char=start,
                        end_char=end,
                        chunk_index=chunk_index,
                    ),
                )
            )
            chunk_index += 1
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy.

        Args:
            text: Text to split.
            separators: Remaining separators to try (first = current, rest = fallbacks).

        Returns:
            List of text segments all ≤ chunk_size characters.
        """
        if not text:
            return []

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            # Base case: split into raw chunk_size character windows
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        result: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(separator)

            if current_len + part_len <= self.chunk_size:
                current.append(part)
                current_len += part_len
            else:
                # Flush accumulated parts as one chunk
                if current:
                    result.append(separator.join(current))
                    current = []
                    current_len = 0

                if len(part) > self.chunk_size and remaining:
                    # Part itself is too large — recurse with smaller separator
                    result.extend(self._split_text(part, remaining))
                else:
                    current.append(part)
                    current_len = part_len

        if current:
            result.append(separator.join(current))

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap by prepending the tail of the previous chunk.

        Args:
            chunks: List of raw text segments from _split_text.

        Returns:
            List of strings where each chunk (except the first) starts with the
            last chunk_overlap characters of the previous chunk.
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks

        merged: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            # Prepend the overlap suffix from the previous chunk
            tail = merged[-1][-self.chunk_overlap:]
            merged.append(tail + chunks[i])
        return merged
