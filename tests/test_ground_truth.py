"""Tests for src/evaluation/ground_truth.py — loading and generation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.evaluation.ground_truth import generate_ground_truth_candidates, load_ground_truth
from src.schemas import (
    Chunk,
    ChunkMetadata,
    GeneratedQAPair,
    GroundTruthChunk,
    GroundTruthSet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "Some product info", chunk_index: int = 0) -> Chunk:
    meta = ChunkMetadata(
        document_id="doc-1",
        source="catalog.pdf",
        page_number=0,
        start_char=0,
        end_char=len(content),
        chunk_index=chunk_index,
    )
    return Chunk(id=str(uuid.uuid4()), content=content, metadata=meta)


def _minimal_ground_truth_dict() -> dict:
    return {
        "queries": [
            {
                "query_id": "q1",
                "question": "What is the return policy?",
                "relevant_chunks": [
                    {"chunk_id": "chunk-abc", "relevance_grade": 3},
                ],
            }
        ]
    }


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_valid_json_loads_successfully(self, tmp_path: Path) -> None:
        gt_file = tmp_path / "ground_truth.json"
        gt_file.write_text(json.dumps(_minimal_ground_truth_dict()), encoding="utf-8")

        result = load_ground_truth(str(gt_file))

        assert isinstance(result, GroundTruthSet)
        assert len(result.queries) == 1
        assert result.queries[0].query_id == "q1"
        assert result.queries[0].relevant_chunks[0].relevance_grade == 3

    def test_multiple_queries_load(self, tmp_path: Path) -> None:
        data = {
            "queries": [
                {
                    "query_id": f"q{i}",
                    "question": f"Question {i}?",
                    "relevant_chunks": [{"chunk_id": f"c{i}", "relevance_grade": 1}],
                }
                for i in range(5)
            ]
        }
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(data), encoding="utf-8")

        result = load_ground_truth(str(gt_file))

        assert len(result.queries) == 5

    def test_missing_required_field_raises_validation_error(self, tmp_path: Path) -> None:
        # Missing 'question' field
        bad = {
            "queries": [
                {
                    "query_id": "q1",
                    # no 'question'
                    "relevant_chunks": [{"chunk_id": "c1", "relevance_grade": 2}],
                }
            ]
        }
        gt_file = tmp_path / "bad.json"
        gt_file.write_text(json.dumps(bad), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_ground_truth(str(gt_file))

    def test_empty_queries_list_raises_validation_error(self, tmp_path: Path) -> None:
        gt_file = tmp_path / "empty.json"
        gt_file.write_text(json.dumps({"queries": []}), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_ground_truth(str(gt_file))

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ground_truth("/nonexistent/path/ground_truth.json")

    def test_invalid_relevance_grade_raises(self, tmp_path: Path) -> None:
        bad = {
            "queries": [
                {
                    "query_id": "q1",
                    "question": "Q?",
                    "relevant_chunks": [{"chunk_id": "c1", "relevance_grade": 99}],
                }
            ]
        }
        gt_file = tmp_path / "bad_grade.json"
        gt_file.write_text(json.dumps(bad), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_ground_truth(str(gt_file))

    def test_empty_relevant_chunks_raises(self, tmp_path: Path) -> None:
        bad = {
            "queries": [
                {
                    "query_id": "q1",
                    "question": "Q?",
                    "relevant_chunks": [],
                }
            ]
        }
        gt_file = tmp_path / "bad_chunks.json"
        gt_file.write_text(json.dumps(bad), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_ground_truth(str(gt_file))


# ---------------------------------------------------------------------------
# generate_ground_truth_candidates
# ---------------------------------------------------------------------------


def _make_generated_pair(question: str = "What is the warranty?") -> GeneratedQAPair:
    return GeneratedQAPair(
        question=question,
        relevant_chunks=[GroundTruthChunk(chunk_id="chunk-1", relevance_grade=3)],
    )


class TestGenerateGroundTruthCandidates:
    def test_empty_chunks_returns_empty_list(self) -> None:
        result = generate_ground_truth_candidates([], n=5)
        assert result == []

    def test_returns_list_of_generated_qa_pairs(self) -> None:
        chunks = [_make_chunk(f"Content {i}", i) for i in range(5)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            result = generate_ground_truth_candidates(chunks, n=1, model="gpt-4o-mini")

        assert len(result) == 1
        assert isinstance(result[0], GeneratedQAPair)
        assert result[0].question == "What is the warranty?"

    def test_batches_chunks_in_groups_of_10(self) -> None:
        chunks = [_make_chunk(f"Chunk {i}", i) for i in range(25)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            # n=3 → 3 batches
            result = generate_ground_truth_candidates(chunks, n=3, model="gpt-4o-mini")

        assert len(result) == 3
        assert mock_client.chat.completions.create.call_count == 3

    def test_n_limits_number_of_pairs(self) -> None:
        # 20 chunks available but n=2 → only 2 batches
        chunks = [_make_chunk(f"Content {i}", i) for i in range(20)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            result = generate_ground_truth_candidates(chunks, n=2, model="gpt-4o-mini")

        assert len(result) == 2

    def test_failed_batch_is_skipped(self) -> None:
        chunks = [_make_chunk(f"Content {i}", i) for i in range(20)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [Exception("API error"), mock_pair]

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            result = generate_ground_truth_candidates(chunks, n=2, model="gpt-4o-mini")

        # First batch failed → skipped; second succeeded → 1 result
        assert len(result) == 1

    def test_instructor_client_created_with_litellm(self) -> None:
        chunks = [_make_chunk("test content", 0)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm") as mock_from_litellm, \
             patch("src.evaluation.ground_truth.litellm") as mock_litellm:
            mock_from_litellm.return_value = mock_client
            generate_ground_truth_candidates(chunks, n=1, model="gpt-4o-mini")

        mock_from_litellm.assert_called_once_with(mock_litellm.completion)

    def test_chunk_ids_appear_in_prompt(self) -> None:
        chunk = _make_chunk("Some content about returns", 0)
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            generate_ground_truth_candidates([chunk], n=1, model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"] if call_kwargs.kwargs else call_kwargs[1]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        # chunk ID must be in the prompt so LLM can reference it
        assert chunk.id in user_content

    def test_model_passed_to_instructor(self) -> None:
        chunks = [_make_chunk("content", 0)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            generate_ground_truth_candidates(chunks, n=1, model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args
        model_arg = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert model_arg == "gpt-4o-mini"

    def test_response_model_is_generated_qa_pair(self) -> None:
        chunks = [_make_chunk("content", 0)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            generate_ground_truth_candidates(chunks, n=1, model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args
        rm = call_kwargs.kwargs.get("response_model") or call_kwargs[1].get("response_model")
        assert rm is GeneratedQAPair

    def test_max_retries_is_3(self) -> None:
        chunks = [_make_chunk("content", 0)]
        mock_pair = _make_generated_pair()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_pair

        with patch("src.evaluation.ground_truth.instructor.from_litellm", return_value=mock_client):
            generate_ground_truth_candidates(chunks, n=1, model="gpt-4o-mini")

        call_kwargs = mock_client.chat.completions.create.call_args
        retries = call_kwargs.kwargs.get("max_retries") or call_kwargs[1].get("max_retries")
        assert retries == 3
