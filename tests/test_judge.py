"""Tests for src/evaluation/judge.py — LLMJudge scoring and batch aggregation."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.cache import JSONCache
from src.evaluation.judge import LLMJudge, _JUDGE_SYSTEM_PROMPT
from src.schemas import Chunk, ChunkMetadata, JudgeResult, JudgeScores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "Product info about warranty") -> Chunk:
    meta = ChunkMetadata(
        document_id="doc-1",
        source="catalog.pdf",
        page_number=0,
        start_char=0,
        end_char=len(content),
        chunk_index=0,
    )
    return Chunk(id=str(uuid.uuid4()), content=content, metadata=meta)


def _make_judge_result(
    relevance: int = 3,
    accuracy: int = 3,
    completeness: int = 3,
    conciseness: int = 3,
    citation_quality: int = 3,
) -> JudgeResult:
    return JudgeResult(
        relevance=relevance,
        accuracy=accuracy,
        completeness=completeness,
        conciseness=conciseness,
        citation_quality=citation_quality,
    )


def _make_mock_client(return_value: JudgeResult | None = None, side_effect=None) -> MagicMock:
    mock = MagicMock()
    if side_effect is not None:
        mock.chat.completions.create.side_effect = side_effect
    else:
        mock.chat.completions.create.return_value = return_value or _make_judge_result()
    return mock


# ---------------------------------------------------------------------------
# LLMJudge.score — basic behaviour
# ---------------------------------------------------------------------------


class TestLLMJudgeScore:
    def test_returns_judge_result(self) -> None:
        expected = _make_judge_result(relevance=4, accuracy=5)
        mock_client = _make_mock_client(expected)

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            result = judge.score("What is the return policy?", "30-day returns.", [])

        assert isinstance(result, JudgeResult)
        assert result.relevance == 4
        assert result.accuracy == 5

    def test_model_passed_to_instructor(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("Q?", "A.", [])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    def test_response_model_is_judge_result(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("Q?", "A.", [])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_model"] is JudgeResult

    def test_max_retries_is_3(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("Q?", "A.", [])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_retries"] == 3

    def test_system_prompt_in_messages(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("Q?", "A.", [])

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        system_content = next(m["content"] for m in messages if m["role"] == "system")
        assert system_content == _JUDGE_SYSTEM_PROMPT

    def test_chunk_content_in_user_prompt(self) -> None:
        chunk = _make_chunk("important warranty detail here")
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("Q?", "A.", [chunk])

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "important warranty detail here" in user_content

    def test_query_and_answer_in_user_prompt(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score("What is the SKU?", "The SKU is ABC-123.", [])

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "What is the SKU?" in user_content
        assert "The SKU is ABC-123." in user_content

    def test_instructor_created_with_litellm_completion(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm") as mock_from_litellm, \
             patch("src.evaluation.judge.litellm") as mock_litellm:
            mock_from_litellm.return_value = mock_client
            LLMJudge(model="gpt-4o-mini")

        mock_from_litellm.assert_called_once_with(mock_litellm.completion)


# ---------------------------------------------------------------------------
# LLMJudge.score — cache behaviour
# ---------------------------------------------------------------------------


class TestLLMJudgeScoreCache:
    def test_cache_hit_skips_llm_call(self) -> None:
        cached_data = _make_judge_result(relevance=5).model_dump()
        mock_client = _make_mock_client()
        mock_cache = MagicMock(spec=JSONCache)
        mock_cache.make_key.return_value = "key-abc"
        mock_cache.get.return_value = cached_data

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini", cache=mock_cache)
            result = judge.score("Q?", "A.", [])

        mock_client.chat.completions.create.assert_not_called()
        assert result.relevance == 5

    def test_cache_miss_calls_llm_and_stores_result(self) -> None:
        expected = _make_judge_result(accuracy=4)
        mock_client = _make_mock_client(expected)
        mock_cache = MagicMock(spec=JSONCache)
        mock_cache.make_key.return_value = "key-xyz"
        mock_cache.get.return_value = None  # cache miss

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini", cache=mock_cache)
            result = judge.score("Q?", "A.", [])

        mock_client.chat.completions.create.assert_called_once()
        mock_cache.set.assert_called_once_with("key-xyz", expected.model_dump())
        assert result.accuracy == 4

    def test_no_cache_calls_llm_without_cache_ops(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini", cache=None)
            judge.score("Q?", "A.", [])

        mock_client.chat.completions.create.assert_called_once()

    def test_cache_key_uses_model_and_prompts(self) -> None:
        mock_client = _make_mock_client()
        mock_cache = MagicMock(spec=JSONCache)
        mock_cache.make_key.return_value = "some-key"
        mock_cache.get.return_value = None

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="my-model", cache=mock_cache)
            judge.score("Q?", "A.", [])

        mock_cache.make_key.assert_called_once()
        call_args = mock_cache.make_key.call_args
        assert call_args.args[0] == "my-model"
        assert call_args.args[1] == _JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# LLMJudge.score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_score_called_once_per_pair(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score_batch([
                {"query": "Q1?", "answer": "A1", "chunks": []},
                {"query": "Q2?", "answer": "A2", "chunks": []},
                {"query": "Q3?", "answer": "A3", "chunks": []},
            ])

        assert mock_client.chat.completions.create.call_count == 3

    def test_returns_judge_scores(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            result = judge.score_batch([{"query": "Q?", "answer": "A", "chunks": []}])

        assert isinstance(result, JudgeScores)

    def test_axis_averages_correct(self) -> None:
        r1 = _make_judge_result(relevance=4, accuracy=4, completeness=4, conciseness=4, citation_quality=4)
        r2 = _make_judge_result(relevance=2, accuracy=2, completeness=2, conciseness=2, citation_quality=2)
        mock_client = _make_mock_client(side_effect=[r1, r2])

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            scores = judge.score_batch([
                {"query": "Q1?", "answer": "A1", "chunks": []},
                {"query": "Q2?", "answer": "A2", "chunks": []},
            ])

        assert abs(scores.avg_relevance - 3.0) < 1e-9
        assert abs(scores.avg_accuracy - 3.0) < 1e-9
        assert abs(scores.avg_completeness - 3.0) < 1e-9
        assert abs(scores.avg_conciseness - 3.0) < 1e-9
        assert abs(scores.avg_citation_quality - 3.0) < 1e-9

    def test_overall_average_is_mean_of_5_axes(self) -> None:
        r = _make_judge_result(relevance=5, accuracy=3, completeness=4, conciseness=2, citation_quality=1)
        expected_overall = (5 + 3 + 4 + 2 + 1) / 5
        mock_client = _make_mock_client(r)

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            scores = judge.score_batch([{"query": "Q?", "answer": "A", "chunks": []}])

        assert abs(scores.overall_average - expected_overall) < 1e-9

    def test_overall_matches_hand_computed(self) -> None:
        # 3 pairs; verify overall_average == mean of per-axis averages
        results = [
            _make_judge_result(relevance=5, accuracy=4, completeness=3, conciseness=2, citation_quality=1),
            _make_judge_result(relevance=1, accuracy=2, completeness=3, conciseness=4, citation_quality=5),
        ]
        mock_client = _make_mock_client(side_effect=results)

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            scores = judge.score_batch([
                {"query": "Q1?", "answer": "A1", "chunks": []},
                {"query": "Q2?", "answer": "A2", "chunks": []},
            ])

        # Each axis avg = (5+1)/2=3, (4+2)/2=3, (3+3)/2=3, (2+4)/2=3, (1+5)/2=3
        expected_overall = (3 + 3 + 3 + 3 + 3) / 5
        assert abs(scores.overall_average - expected_overall) < 1e-9

    def test_chunks_passed_to_score(self) -> None:
        chunk = _make_chunk("return policy details")
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            judge.score_batch([{"query": "Q?", "answer": "A", "chunks": [chunk]}])

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "return policy details" in user_content

    def test_missing_chunks_key_defaults_to_empty(self) -> None:
        mock_client = _make_mock_client()

        with patch("src.evaluation.judge.instructor.from_litellm", return_value=mock_client):
            judge = LLMJudge(model="gpt-4o-mini")
            # No 'chunks' key in the pair dict
            scores = judge.score_batch([{"query": "Q?", "answer": "A"}])

        assert isinstance(scores, JudgeScores)
