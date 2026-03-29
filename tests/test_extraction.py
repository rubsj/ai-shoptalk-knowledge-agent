"""Tests for src/extraction.py.

Structure:
  TestCleanText           — unit tests, no PDFs required
  TestIsHeaderOrFooter    — unit tests, no PDFs required
  TestRemoveHeadersFooters— unit tests, no PDFs required
  TestExtractPdf          — integration tests, requires PDFs in data/pdfs/
  TestExtractAllPdfs      — integration tests, requires PDFs in data/pdfs/
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

from src.extraction import (
    _describe_page_images,
    _extract_page_content,
    _extract_text_from_dict_block,
    _is_header_or_footer,
    clean_text,
    extract_all_pdfs,
    extract_pdf,
    load_document,
    remove_headers_footers,
    save_document,
)
from src.schemas import Document, DocumentMetadata, PageInfo

# Path to test PDFs (relative to repo root, resolved at runtime)
_PDF_DIR = Path(__file__).parent.parent / "data" / "pdfs"
_PDFS_PRESENT = _PDF_DIR.exists() and len(list(_PDF_DIR.glob("*.pdf"))) > 0

_skip_if_no_pdfs = pytest.mark.skipif(
    not _PDFS_PRESENT,
    reason="PDF files not found in data/pdfs/ — run manually to download",
)

# ---------------------------------------------------------------------------
# TestCleanText
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_ligature_fi(self):
        assert clean_text("\ufb01eld") == "field"

    def test_ligature_fl(self):
        assert clean_text("\ufb02oor") == "floor"

    def test_ligature_ffi(self):
        assert clean_text("\ufb03cient") == "fficient"

    def test_ligature_ff(self):
        assert clean_text("\ufb00ect") == "ffect"

    def test_ligature_ffl(self):
        assert clean_text("\ufb04uent") == "ffluent"

    def test_hyphenation_rejoin(self):
        # "trans-\nformer" → "transformer"
        assert clean_text("trans-\nformer") == "transformer"

    def test_hyphenation_rejoin_multiword(self):
        text = "atten-\ntion is all you need"
        result = clean_text(text)
        assert "attention" in result

    def test_collapse_excessive_newlines(self):
        result = clean_text("para1\n\n\n\npara2")
        assert "\n\n\n" not in result
        assert "para1" in result and "para2" in result

    def test_collapse_excessive_spaces(self):
        result = clean_text("word1   word2")
        assert "word1 word2" in result

    def test_strip_leading_trailing_whitespace(self):
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_no_op_on_clean_text(self):
        text = "This is already clean text."
        assert clean_text(text) == text

    def test_multiple_ligatures_in_text(self):
        # "official" has fi ligature
        result = clean_text("o\ufb03cial results")
        assert result == "official results"


# ---------------------------------------------------------------------------
# TestIsHeaderOrFooter
# ---------------------------------------------------------------------------


class TestIsHeaderOrFooter:
    @pytest.mark.parametrize("line", ["3", "  10  ", "  42  "])
    def test_standalone_digit_is_header(self, line: str):
        assert _is_header_or_footer(line, 0, 10) is True

    @pytest.mark.parametrize("line", ["Page 3", "page 10", "Page 3 of 15"])
    def test_page_n_is_header(self, line: str):
        assert _is_header_or_footer(line, 0, 10) is True

    @pytest.mark.parametrize("line", ["3 of 10", "1 of 5"])
    def test_n_of_m_is_header(self, line: str):
        assert _is_header_or_footer(line, 0, 10) is True

    @pytest.mark.parametrize("line", ["arXiv:1706.03762v5", "arXiv:1810.04805"])
    def test_arxiv_is_header(self, line: str):
        assert _is_header_or_footer(line, 0, 10) is True

    def test_short_all_caps_is_header(self):
        assert _is_header_or_footer("NEURAL INFORMATION PROCESSING", 0, 10) is True

    def test_normal_sentence_is_not_header(self):
        assert _is_header_or_footer("The attention mechanism is a key component.", 0, 10) is False

    def test_empty_line_is_not_header(self):
        assert _is_header_or_footer("", 0, 10) is False

    def test_whitespace_only_is_not_header(self):
        assert _is_header_or_footer("   ", 0, 10) is False

    def test_long_caps_is_not_header(self):
        # >60 chars — too long to be a header
        long_caps = "A" * 61
        assert _is_header_or_footer(long_caps, 0, 10) is False

    def test_mixed_case_is_not_header(self):
        assert _is_header_or_footer("Introduction", 0, 10) is False


# ---------------------------------------------------------------------------
# TestRemoveHeadersFooters
# ---------------------------------------------------------------------------


class TestRemoveHeadersFooters:
    def _make_page(self, header: str | None = None, footer: str | None = None, body_lines: int = 10) -> str:
        """Build a synthetic page with optional header/footer."""
        lines = []
        if header:
            lines.append(header)
        lines.extend([f"Body line {i} with meaningful content here." for i in range(body_lines)])
        if footer:
            lines.append(footer)
        return "\n".join(lines)

    def test_removes_page_number_header(self):
        text = self._make_page(header="3")
        result = remove_headers_footers(text, 2, 10)
        assert "3" not in result.split("\n")[:2]
        assert "Body line 0" in result

    def test_removes_page_number_footer(self):
        text = self._make_page(footer="3")
        result = remove_headers_footers(text, 2, 10)
        assert "Body line 0" in result
        # "3" should not appear as a standalone line
        lines = result.split("\n")
        assert not any(line.strip() == "3" for line in lines)

    def test_short_page_untouched(self):
        # ≤6 lines — leave as-is
        short = "\n".join(["1", "line 2", "line 3"])
        result = remove_headers_footers(short, 0, 10)
        assert result == short

    def test_body_without_headers_unchanged(self):
        # No header/footer lines — body should come through intact
        text = self._make_page(body_lines=8)
        result = remove_headers_footers(text, 0, 10)
        # All body lines should still be present
        assert "Body line 0" in result
        assert "Body line 7" in result


# ---------------------------------------------------------------------------
# TestExtractPdf (integration — requires real PDFs)
# ---------------------------------------------------------------------------


class TestExtractPdf:
    @_skip_if_no_pdfs
    @pytest.mark.parametrize("filename,min_pages,keyword", [
        ("attention-is-all-you-need.pdf", 10, "attention"),
        ("bert-devlin-et-al.pdf", 10, "BERT"),
        ("rag-lewis-et-al.pdf", 10, "retrieval"),
        ("sentence-bert.pdf", 5, "sentence"),
    ])
    def test_extract_paper(self, filename: str, min_pages: int, keyword: str):
        pdf_path = _PDF_DIR / filename
        doc = extract_pdf(pdf_path)

        assert isinstance(doc, Document)
        assert doc.metadata.page_count >= min_pages
        assert len(doc.pages) == doc.metadata.page_count
        assert doc.metadata.source.endswith(filename)
        assert doc.id  # uuid4 generated
        assert len(doc.content) > 1000  # substantial content extracted

        # Verify keyword appears somewhere (case-insensitive)
        assert keyword.lower() in doc.content.lower(), (
            f"Expected keyword '{keyword}' not found in {filename}"
        )

    @_skip_if_no_pdfs
    def test_pages_have_content(self):
        pdf_path = _PDF_DIR / "attention-is-all-you-need.pdf"
        doc = extract_pdf(pdf_path)
        non_empty = [p for p in doc.pages if p.char_count > 0]
        # Most pages should have content (allow a few image-only pages)
        assert len(non_empty) > doc.metadata.page_count * 0.7

    @_skip_if_no_pdfs
    def test_page_numbers_sequential(self):
        pdf_path = _PDF_DIR / "attention-is-all-you-need.pdf"
        doc = extract_pdf(pdf_path)
        for i, page in enumerate(doc.pages):
            assert page.page_number == i

    @_skip_if_no_pdfs
    def test_char_count_matches_text_length(self):
        pdf_path = _PDF_DIR / "attention-is-all-you-need.pdf"
        doc = extract_pdf(pdf_path)
        for page in doc.pages:
            assert page.char_count == len(page.text)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_pdf(tmp_path / "nonexistent.pdf")


# ---------------------------------------------------------------------------
# TestExtractAllPdfs (integration)
# ---------------------------------------------------------------------------


class TestExtractAllPdfs:
    @_skip_if_no_pdfs
    def test_extracts_all_four_papers(self):
        docs = extract_all_pdfs(_PDF_DIR)
        assert len(docs) == 4

    @_skip_if_no_pdfs
    def test_returns_document_objects(self):
        docs = extract_all_pdfs(_PDF_DIR)
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.content

    def test_empty_dir_returns_empty_list(self, tmp_path: Path):
        docs = extract_all_pdfs(tmp_path)
        assert docs == []


# ---------------------------------------------------------------------------
# TestRemoveHeadersFooters — edge cases for else branches (lines 124-134)
# ---------------------------------------------------------------------------


class TestRemoveHeadersFootersEdgeCases:
    def test_all_three_header_lines_are_headers(self):
        """When all 3 header lines match, the for-else branch fires (line 125)."""
        lines = ["3", "Page 5", "arXiv:1234"]  # all headers
        lines += [f"Body line {i} with content." for i in range(10)]
        text = "\n".join(lines)
        result = remove_headers_footers(text, 0, 10)
        assert "Body line 0" in result
        assert "3\n" not in result.split("\n")[0]

    def test_all_three_footer_lines_are_footers(self):
        """When all 3 footer lines match, the for-else branch fires (line 134)."""
        lines = [f"Body line {i} with content." for i in range(10)]
        lines += ["42", "Page 10", "arXiv:9999"]  # all footers
        text = "\n".join(lines)
        result = remove_headers_footers(text, 9, 10)
        assert "Body line 0" in result
        assert "42" not in result.split("\n")[-1]

    def test_all_header_and_footer_lines_are_headers(self):
        """Both head and foot for-else branches fire."""
        lines = ["1", "Page 1", "NEURAL INFO"]
        lines += [f"Body {i}" for i in range(10)]
        lines += ["99", "Page 99", "arXiv:0000"]
        text = "\n".join(lines)
        result = remove_headers_footers(text, 0, 100)
        assert "Body 0" in result
        assert "Body 9" in result


# ---------------------------------------------------------------------------
# TestDescribePageImages (mocked vision LLM, lines 163-188)
# ---------------------------------------------------------------------------


class TestDescribePageImages:
    def test_returns_description_when_visuals_found(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Figure 1 shows a transformer architecture."
        with patch("src.extraction.litellm.completion", return_value=mock_response):
            result = _describe_page_images(b"\x89PNG\r\n", page_number=0)
        assert "Figure 1" in result

    def test_returns_empty_when_no_visuals(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "NO_VISUAL_ELEMENTS"
        with patch("src.extraction.litellm.completion", return_value=mock_response):
            result = _describe_page_images(b"\x89PNG\r\n", page_number=0)
        assert result == ""


# ---------------------------------------------------------------------------
# TestExtractTextFromDictBlock (lines 197-202)
# ---------------------------------------------------------------------------


class TestExtractTextFromDictBlock:
    def test_extracts_text_from_block(self):
        block = {
            "lines": [
                {"spans": [{"text": "Hello "}, {"text": "world"}]},
                {"spans": [{"text": "Second line"}]},
            ]
        }
        result = _extract_text_from_dict_block(block)
        assert "Hello world" in result
        assert "Second line" in result

    def test_empty_lines(self):
        block = {"lines": []}
        assert _extract_text_from_dict_block(block) == ""

    def test_skips_whitespace_only_spans(self):
        block = {"lines": [{"spans": [{"text": "   "}]}]}
        result = _extract_text_from_dict_block(block)
        assert result == ""


# ---------------------------------------------------------------------------
# TestExtractPageContent with describe_images=True (lines 231-268)
# ---------------------------------------------------------------------------


class TestExtractPageContentWithImages:
    def _make_mock_page(self, has_images: bool = True):
        page = MagicMock()

        text_block = {
            "type": 0,
            "bbox": [72, 100, 500, 200],
            "lines": [{"spans": [{"text": "Some text content"}]}],
        }
        blocks = [text_block]
        if has_images:
            image_block = {"type": 1, "bbox": [72, 250, 500, 400]}
            blocks.append(image_block)

        page.get_text.side_effect = lambda fmt=None: (
            {"blocks": blocks} if fmt == "dict" else "Some text content"
        )
        pixmap = MagicMock()
        pixmap.tobytes.return_value = b"\x89PNG\r\n"
        page.get_pixmap.return_value = pixmap
        return page

    def test_describe_images_true_with_image_blocks(self):
        page = self._make_mock_page(has_images=True)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Figure 1: attention scores"
        with patch("src.extraction.litellm.completion", return_value=mock_response):
            result = _extract_page_content(page, 0, 10, describe_images=True, vision_model="gpt-4o-mini")
        assert "Visual Content" in result
        assert "attention scores" in result

    def test_describe_images_true_no_image_blocks(self):
        page = self._make_mock_page(has_images=False)
        result = _extract_page_content(page, 0, 10, describe_images=True, vision_model="gpt-4o-mini")
        assert "Visual Content" not in result

    def test_describe_images_false(self):
        page = MagicMock()
        page.get_text.return_value = "Plain text from page."
        result = _extract_page_content(page, 0, 10, describe_images=False, vision_model="gpt-4o-mini")
        assert "Plain text from page" in result

    def test_vision_exception_handled_gracefully(self):
        page = self._make_mock_page(has_images=True)
        with patch("src.extraction.litellm.completion", side_effect=RuntimeError("API down")):
            result = _extract_page_content(page, 0, 10, describe_images=True, vision_model="gpt-4o-mini")
        assert "Some text content" in result
        assert "Visual Content" not in result


# ---------------------------------------------------------------------------
# TestSaveLoadDocument (lines 332-338, 345)
# ---------------------------------------------------------------------------


class TestSaveLoadDocument:
    def _make_doc(self) -> Document:
        return Document(
            content="Test content for save/load.",
            metadata=DocumentMetadata(source="test.pdf", title="Test", author="Author", page_count=1),
            pages=[PageInfo(page_number=0, text="Test content for save/load.", char_count=27)],
        )

    def test_save_creates_file(self, tmp_path: Path):
        doc = self._make_doc()
        out = save_document(doc, tmp_path / "doc.json")
        assert out.exists()

    def test_save_load_roundtrip(self, tmp_path: Path):
        doc = self._make_doc()
        path = save_document(doc, tmp_path / "doc.json")
        loaded = load_document(path)
        assert loaded is not None
        assert loaded.content == doc.content
        assert loaded.metadata.source == doc.metadata.source

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        result = load_document(tmp_path / "nope.json")
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        doc = self._make_doc()
        path = save_document(doc, tmp_path / "sub" / "dir" / "doc.json")
        assert path.exists()


# ---------------------------------------------------------------------------
# TestExtractPdfEmptyContent (line 313)
# ---------------------------------------------------------------------------


class TestExtractPdfEmptyContent:
    def test_empty_pdf_raises_value_error(self, tmp_path: Path):
        """A PDF with no extractable text should raise ValueError."""
        import fitz
        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()
        with pytest.raises(ValueError, match="No extractable text"):
            extract_pdf(pdf_path)


# ---------------------------------------------------------------------------
# TestExtractAllPdfsCacheMiss (lines 387-397)
# ---------------------------------------------------------------------------


class TestExtractAllPdfsCacheMiss:
    @_skip_if_no_pdfs
    def test_cache_miss_extracts_and_saves(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        docs = extract_all_pdfs(_PDF_DIR, cache_dir=cache_dir)
        assert len(docs) == 4
        # Check JSON cache files were created
        cached_files = list(cache_dir.glob("*.json"))
        assert len(cached_files) == 4
        # Check validation .txt files
        validation_files = list((cache_dir / "validation").glob("*.txt"))
        assert len(validation_files) == 4

    @_skip_if_no_pdfs
    def test_cache_hit_skips_extraction(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        # First run: cache miss
        extract_all_pdfs(_PDF_DIR, cache_dir=cache_dir)
        # Second run: cache hit
        docs = extract_all_pdfs(_PDF_DIR, cache_dir=cache_dir)
        assert len(docs) == 4
