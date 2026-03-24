"""Tests for LiteLLMClient, build_qa_prompt, and extract_citations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.cache import JSONCache
from src.generator import LiteLLMClient, build_qa_prompt, extract_citations
from src.interfaces import BaseLLM
from src.schemas import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "hello world", idx: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="doc1",
            source="test.pdf",
            page_number=idx,
            start_char=0,
            end_char=len(content),
            chunk_index=idx,
        ),
    )


def _mock_completion(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# LiteLLMClient — basic generation
# ---------------------------------------------------------------------------


class TestLiteLLMClient:
    def test_implements_base_llm(self) -> None:
        client = LiteLLMClient()
        assert isinstance(client, BaseLLM)

    def test_returns_content_string(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("answer")):
            client = LiteLLMClient()
            result = client.generate("question")
        assert result == "answer"

    def test_system_prompt_included_when_non_empty(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("ok")) as mock_comp:
            client = LiteLLMClient()
            client.generate("q", system_prompt="You are helpful.")
        messages = mock_comp.call_args[1]["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "q"}

    def test_system_prompt_omitted_when_empty(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("ok")) as mock_comp:
            client = LiteLLMClient()
            client.generate("q", system_prompt="")
        messages = mock_comp.call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_temperature_forwarded(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("ok")) as mock_comp:
            client = LiteLLMClient()
            client.generate("q", temperature=0.7)
        assert mock_comp.call_args[1]["temperature"] == 0.7

    def test_model_forwarded(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("ok")) as mock_comp:
            client = LiteLLMClient(model="gpt-4o")
            client.generate("q")
        assert mock_comp.call_args[1]["model"] == "gpt-4o"

    # ------------------------------------------------------------------
    # Cache integration
    # ------------------------------------------------------------------

    def test_generate_works_without_cache(self) -> None:
        with patch("src.generator.litellm.completion", return_value=_mock_completion("no cache")):
            client = LiteLLMClient(cache=None)
            result = client.generate("q")
        assert result == "no cache"

    def test_generate_populates_cache_on_miss(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        with patch("src.generator.litellm.completion", return_value=_mock_completion("fresh")):
            client = LiteLLMClient(cache=cache)
            client.generate("q", system_prompt="sys")
        key = cache.make_key("gpt-4o-mini", "sys", "q")
        assert cache.get(key) == {"content": "fresh"}

    def test_generate_uses_cache_on_hit(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("gpt-4o-mini", "sys", "q")
        cache.set(key, {"content": "cached answer"})

        with patch("src.generator.litellm.completion") as mock_comp:
            client = LiteLLMClient(cache=cache)
            result = client.generate("q", system_prompt="sys")

        mock_comp.assert_not_called()
        assert result == "cached answer"

    def test_cache_key_includes_model(self, tmp_path: pytest.TempPathFactory) -> None:
        """Different models must not share cached responses."""
        cache = JSONCache(str(tmp_path))
        key = cache.make_key("gpt-4o", "sys", "q")
        cache.set(key, {"content": "gpt4o answer"})

        with patch("src.generator.litellm.completion", return_value=_mock_completion("mini answer")) as mock_comp:
            client = LiteLLMClient(model="gpt-4o-mini", cache=cache)
            result = client.generate("q", system_prompt="sys")

        mock_comp.assert_called_once()
        assert result == "mini answer"


# ---------------------------------------------------------------------------
# build_qa_prompt
# ---------------------------------------------------------------------------


class TestBuildQaPrompt:
    def test_contains_query(self) -> None:
        chunks = [_make_chunk("some context")]
        prompt = build_qa_prompt("What is RAG?", chunks)
        assert "What is RAG?" in prompt

    def test_numbers_chunks_from_one(self) -> None:
        chunks = [_make_chunk("first"), _make_chunk("second", idx=1)]
        prompt = build_qa_prompt("q", chunks)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_chunk_content_included(self) -> None:
        chunks = [_make_chunk("unique content XYZ")]
        prompt = build_qa_prompt("q", chunks)
        assert "unique content XYZ" in prompt

    def test_cite_instruction_present(self) -> None:
        chunks = [_make_chunk("ctx")]
        prompt = build_qa_prompt("q", chunks)
        assert "[N]" in prompt or "cite" in prompt.lower()

    def test_multiple_chunks_all_present(self) -> None:
        chunks = [_make_chunk(f"doc{i}", idx=i) for i in range(5)]
        prompt = build_qa_prompt("q", chunks)
        for i in range(5):
            assert f"doc{i}" in prompt


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_valid_citation_parsed(self) -> None:
        chunks = [_make_chunk("first chunk")]
        citations = extract_citations("See [1] for details.", chunks)
        assert len(citations) == 1
        assert citations[0].chunk_id == chunks[0].id

    def test_citation_fields_populated(self) -> None:
        chunk = _make_chunk("some text", idx=2)
        citations = extract_citations("[1] answers this.", [chunk])
        c = citations[0]
        assert c.source == chunk.metadata.source
        assert c.page_number == chunk.metadata.page_number
        assert c.text_snippet == chunk.content[:100]
        assert c.relevance_score == 0.0

    def test_out_of_range_citation_skipped(self) -> None:
        chunks = [_make_chunk("only one")]
        citations = extract_citations("[99] is invalid.", chunks)
        assert citations == []

    def test_zero_citation_skipped(self) -> None:
        chunks = [_make_chunk("chunk")]
        citations = extract_citations("[0] is invalid.", chunks)
        assert citations == []

    def test_duplicate_citations_deduplicated(self) -> None:
        chunks = [_make_chunk("chunk")]
        citations = extract_citations("[1] and [1] again.", chunks)
        assert len(citations) == 1

    def test_no_markers_returns_empty(self) -> None:
        chunks = [_make_chunk("chunk")]
        citations = extract_citations("No citations here.", chunks)
        assert citations == []

    def test_multiple_valid_citations(self) -> None:
        chunks = [_make_chunk(f"c{i}", idx=i) for i in range(3)]
        citations = extract_citations("See [1], [2], and [3].", chunks)
        assert len(citations) == 3

    def test_mixed_valid_and_invalid(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2", idx=1)]
        citations = extract_citations("[1] valid, [5] invalid, [2] valid.", chunks)
        assert len(citations) == 2

    def test_citation_order_follows_appearance(self) -> None:
        chunks = [_make_chunk(f"c{i}", idx=i) for i in range(3)]
        citations = extract_citations("[3] then [1] then [2].", chunks)
        assert [c.chunk_id for c in citations] == [
            chunks[2].id, chunks[0].id, chunks[1].id
        ]

    def test_empty_chunks_list_skips_all(self) -> None:
        citations = extract_citations("[1] reference.", [])
        assert citations == []
