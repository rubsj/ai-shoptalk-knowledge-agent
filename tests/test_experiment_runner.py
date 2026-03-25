"""Tests for src/experiment_runner.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.experiment_runner import (
    _EMBEDDER_ORDER,
    _group_configs_by_embedder,
    _run_single_config,
    run_experiment_grid,
)
from src.schemas import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    ExperimentConfig,
    ExperimentResult,
    GroundTruthChunk,
    GroundTruthQuery,
    GroundTruthSet,
    JudgeResult,
    JudgeScores,
    PageInfo,
    PerformanceMetrics,
    RetrievalMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25_config(**overrides) -> ExperimentConfig:
    defaults = dict(chunking_strategy="fixed", retriever_type="bm25")
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _dense_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        chunking_strategy="fixed", retriever_type="dense", embedding_model="minilm"
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _hybrid_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        chunking_strategy="fixed",
        retriever_type="hybrid",
        embedding_model="minilm",
        hybrid_alpha=0.7,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_document() -> Document:
    return Document(
        content="Test document content for evaluation",
        metadata=DocumentMetadata(source="test.pdf", page_count=1),
        pages=[PageInfo(page_number=0, text="Test document content for evaluation", char_count=38)],
    )


def _make_ground_truth(n_queries: int = 1) -> GroundTruthSet:
    queries = [
        GroundTruthQuery(
            query_id=f"q{i}",
            question=f"Question {i}?",
            relevant_chunks=[GroundTruthChunk(chunk_id=f"chunk{i}", relevance_grade=3)],
        )
        for i in range(n_queries)
    ]
    return GroundTruthSet(queries=queries)


def _make_chunk(chunk_id: str = "chunk1") -> Chunk:
    return Chunk(
        id=chunk_id,
        content="chunk content",
        metadata=ChunkMetadata(
            document_id="doc1",
            source="test.pdf",
            page_number=0,
            start_char=0,
            end_char=13,
            chunk_index=0,
        ),
    )


def _make_retrieval_result(chunk_id: str = "chunk1"):
    from src.schemas import RetrievalResult
    return RetrievalResult(
        chunk=_make_chunk(chunk_id),
        score=0.9,
        retriever_type="dense",
        rank=1,
    )


def _make_judge_result() -> JudgeResult:
    return JudgeResult(
        relevance=4, accuracy=4, completeness=4, conciseness=4, citation_quality=4
    )


def _make_judge_scores() -> JudgeScores:
    return JudgeScores(
        avg_relevance=4.0,
        avg_accuracy=4.0,
        avg_completeness=4.0,
        avg_conciseness=4.0,
        avg_citation_quality=4.0,
        overall_average=4.0,
    )


def _make_mock_result(experiment_id: str = "test-id") -> MagicMock:
    """Create a mock ExperimentResult with experiment_id and model_dump."""
    mock = MagicMock(spec=ExperimentResult)
    mock.experiment_id = experiment_id
    mock.model_dump.return_value = {"experiment_id": experiment_id}
    return mock


# ---------------------------------------------------------------------------
# _group_configs_by_embedder
# ---------------------------------------------------------------------------


class TestGroupConfigsByEmbedder:
    def test_bm25_goes_to_none_group(self) -> None:
        config = _bm25_config()
        groups = _group_configs_by_embedder([config])
        assert None in groups
        assert config in groups[None]

    def test_dense_minilm_goes_to_minilm_group(self) -> None:
        config = _dense_config(embedding_model="minilm")
        groups = _group_configs_by_embedder([config])
        assert "minilm" in groups
        assert config in groups["minilm"]

    def test_dense_mpnet_goes_to_mpnet_group(self) -> None:
        config = _dense_config(embedding_model="mpnet")
        groups = _group_configs_by_embedder([config])
        assert "mpnet" in groups

    def test_multiple_configs_same_embedder_in_same_group(self) -> None:
        c1 = _dense_config(chunking_strategy="fixed")
        c2 = _dense_config(chunking_strategy="recursive")
        groups = _group_configs_by_embedder([c1, c2])
        assert len(groups["minilm"]) == 2

    def test_configs_split_across_groups(self) -> None:
        bm25 = _bm25_config()
        dense = _dense_config()
        hybrid = _hybrid_config()
        groups = _group_configs_by_embedder([bm25, dense, hybrid])
        assert len(groups[None]) == 1
        assert len(groups["minilm"]) == 2  # dense + hybrid both minilm

    def test_empty_input_returns_empty_dict(self) -> None:
        groups = _group_configs_by_embedder([])
        assert groups == {}

    def test_openai_group(self) -> None:
        config = _dense_config(embedding_model="openai")
        groups = _group_configs_by_embedder([config])
        assert "openai" in groups


# ---------------------------------------------------------------------------
# _run_single_config
# ---------------------------------------------------------------------------


class TestRunSingleConfig:
    def _mock_all_deps(self, chunk_id: str = "chunk1", answer: str = "Test answer."):
        """Return a context-manager-style patcher dict for _run_single_config deps."""
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [_make_chunk(chunk_id)]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_make_retrieval_result(chunk_id)]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = answer

        return mock_chunker, mock_retriever, mock_llm

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_returns_experiment_result(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer text"

        config = _dense_config()
        mock_embedder = MagicMock()
        mock_embedder.dimensions = 384
        mock_embedder.embed.return_value = MagicMock()

        result = _run_single_config(
            config=config,
            embedder=mock_embedder,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        assert isinstance(result, ExperimentResult)
        assert result.config is config
        assert isinstance(result.metrics, RetrievalMetrics)
        assert isinstance(result.performance, PerformanceMetrics)

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_bm25_config_skips_vector_store(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        config = _bm25_config()
        result = _run_single_config(
            config=config,
            embedder=None,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        mock_faiss.assert_not_called()
        assert result.performance.embedding_source == "none"
        assert result.performance.index_size_bytes == 0
        assert result.performance.cost_estimate_usd == 0.0

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_judge_called_per_query(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        mock_judge = MagicMock()
        mock_judge.score.return_value = _make_judge_result()

        config = _dense_config()
        mock_embedder = MagicMock()
        mock_embedder.dimensions = 384
        mock_embedder.embed.return_value = MagicMock()

        gt = _make_ground_truth(n_queries=2)
        result = _run_single_config(
            config=config,
            embedder=mock_embedder,
            documents=[_make_document()],
            ground_truth=gt,
            judge=mock_judge,
            cache=None,
        )

        # judge.score called once per query, not score_batch
        assert mock_judge.score.call_count == 2
        assert result.judge_scores is not None
        # Per-query judge_result populated
        for qr in result.query_results:
            assert qr.judge_result is not None
            assert isinstance(qr.judge_result, JudgeResult)

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_judge_none_when_not_provided(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        config = _bm25_config()
        result = _run_single_config(
            config=config,
            embedder=None,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        assert result.judge_scores is None
        assert result.query_results[0].judge_result is None

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_query_results_match_ground_truth_queries(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        gt = _make_ground_truth(n_queries=3)
        config = _bm25_config()
        result = _run_single_config(
            config=config,
            embedder=None,
            documents=[_make_document()],
            ground_truth=gt,
            judge=None,
            cache=None,
        )

        assert len(result.query_results) == 3
        query_ids = {qr.query_id for qr in result.query_results}
        assert query_ids == {"q0", "q1", "q2"}

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_openai_embedding_source_is_api_with_cost(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        config = _dense_config(embedding_model="openai")
        mock_embedder = MagicMock()
        mock_embedder.dimensions = 1536
        mock_embedder.embed.return_value = MagicMock()

        result = _run_single_config(
            config=config,
            embedder=mock_embedder,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        assert result.performance.embedding_source == "api"
        # 1 chunk × 150 avg tokens × $0.02/1M tokens = 0.000003
        assert result.performance.cost_estimate_usd > 0.0
        expected_cost = (1 * 150 / 1_000_000) * 0.02
        assert abs(result.performance.cost_estimate_usd - expected_cost) < 1e-12

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_local_embedding_source_for_minilm(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        config = _dense_config(embedding_model="minilm")
        mock_embedder = MagicMock()
        mock_embedder.dimensions = 384
        mock_embedder.embed.return_value = MagicMock()

        result = _run_single_config(
            config=config,
            embedder=mock_embedder,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        assert result.performance.embedding_source == "local"
        assert result.performance.cost_estimate_usd == 0.0

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_experiment_id_is_unique_per_call(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        config = _bm25_config()
        r1 = _run_single_config(
            config=config,
            embedder=None,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )
        r2 = _run_single_config(
            config=config,
            embedder=None,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(),
            judge=None,
            cache=None,
        )

        assert r1.experiment_id != r2.experiment_id

    @patch("src.experiment_runner.psutil")
    @patch("src.experiment_runner.FAISSVectorStore")
    @patch("src.experiment_runner.create_retriever")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.create_chunker")
    @patch("src.experiment_runner.build_qa_prompt", return_value="prompt")
    @patch("src.experiment_runner.extract_citations", return_value=[])
    def test_judge_scores_aggregated_from_per_query(
        self,
        mock_extract,
        mock_prompt,
        mock_create_chunker,
        mock_create_llm,
        mock_create_retriever,
        mock_faiss,
        mock_psutil,
    ) -> None:
        mock_psutil.Process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_create_chunker.return_value.chunk.return_value = [_make_chunk()]
        mock_create_retriever.return_value.retrieve.return_value = [_make_retrieval_result()]
        mock_create_llm.return_value.generate.return_value = "answer"

        # Return different scores for each query
        judge_r1 = JudgeResult(relevance=5, accuracy=4, completeness=3, conciseness=4, citation_quality=5)
        judge_r2 = JudgeResult(relevance=3, accuracy=4, completeness=5, conciseness=4, citation_quality=3)
        mock_judge = MagicMock()
        mock_judge.score.side_effect = [judge_r1, judge_r2]

        config = _dense_config()
        mock_embedder = MagicMock()
        mock_embedder.dimensions = 384
        mock_embedder.embed.return_value = MagicMock()

        result = _run_single_config(
            config=config,
            embedder=mock_embedder,
            documents=[_make_document()],
            ground_truth=_make_ground_truth(n_queries=2),
            judge=mock_judge,
            cache=None,
        )

        assert result.judge_scores is not None
        assert result.judge_scores.avg_relevance == 4.0  # (5+3)/2
        assert result.judge_scores.avg_accuracy == 4.0   # (4+4)/2
        assert result.judge_scores.avg_completeness == 4.0  # (3+5)/2
        assert result.judge_scores.avg_conciseness == 4.0   # (4+4)/2
        assert result.judge_scores.avg_citation_quality == 4.0  # (5+3)/2
        assert result.judge_scores.overall_average == 4.0


# ---------------------------------------------------------------------------
# run_experiment_grid
# ---------------------------------------------------------------------------


class TestRunExperimentGrid:
    def _make_configs(self) -> list[ExperimentConfig]:
        return [
            _bm25_config(),
            _dense_config(embedding_model="minilm"),
        ]

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_summary_json_saved(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [_bm25_config()]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result("exp-001")

        out_dir = str(tmp_path / "results")
        run_experiment_grid(
            config_dir="experiments/configs",
            ground_truth_path="data/ground_truth.json",
            output_dir=out_dir,
            documents=[],
            run_judge=False,
        )

        # summary.json written
        summary_path = tmp_path / "results" / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_per_experiment_json_files_written(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [_bm25_config(), _bm25_config()]
        mock_load_gt.return_value = _make_ground_truth()

        mock_r1 = _make_mock_result("exp-001")
        mock_r2 = _make_mock_result("exp-002")
        mock_run_single.side_effect = [mock_r1, mock_r2]

        out_dir = str(tmp_path / "results")
        run_experiment_grid(output_dir=out_dir, run_judge=False)

        assert (tmp_path / "results" / "exp-001.json").exists()
        assert (tmp_path / "results" / "exp-002.json").exists()
        assert (tmp_path / "results" / "summary.json").exists()

        # Each per-experiment file has a single result
        data1 = json.loads((tmp_path / "results" / "exp-001.json").read_text())
        assert data1["experiment_id"] == "exp-001"

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_embedder_loaded_once_per_group(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        # 2 minilm configs — create_embedder("minilm") called once
        c1 = _dense_config(chunking_strategy="fixed")
        c2 = _dense_config(chunking_strategy="recursive")
        mock_load_configs.return_value = [c1, c2]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result()

        out_dir = str(tmp_path / "out")
        run_experiment_grid(output_dir=out_dir, run_judge=False)

        assert mock_create_embedder.call_count == 1
        mock_create_embedder.assert_called_once_with("minilm")

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_gc_collect_called_between_groups(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [
            _dense_config(embedding_model="minilm"),
            _dense_config(embedding_model="mpnet"),
        ]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result()

        out_dir = str(tmp_path / "out")
        run_experiment_grid(output_dir=out_dir, run_judge=False)

        # gc.collect called once per group (minilm + mpnet = 2)
        assert mock_gc.collect.call_count == 2

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_embedder_order_is_minilm_mpnet_none_openai(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [
            _dense_config(embedding_model="openai"),
            _dense_config(embedding_model="minilm"),
            _dense_config(embedding_model="mpnet"),
            _bm25_config(),
        ]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result()

        out_dir = str(tmp_path / "out")
        run_experiment_grid(output_dir=out_dir, run_judge=False)

        call_args = [c.args[0] for c in mock_create_embedder.call_args_list]
        # None group (bm25) doesn't call create_embedder
        assert call_args == ["minilm", "mpnet", "openai"]

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_run_judge_false_skips_judge(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [_bm25_config()]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result()

        out_dir = str(tmp_path / "out")
        run_experiment_grid(output_dir=out_dir, run_judge=False)

        mock_judge_cls.assert_not_called()
        _, kwargs = mock_run_single.call_args
        assert kwargs["judge"] is None

    @patch("src.experiment_runner.gc")
    @patch("src.experiment_runner._run_single_config")
    @patch("src.experiment_runner.create_embedder")
    @patch("src.experiment_runner.LLMJudge")
    @patch("src.experiment_runner.create_llm")
    @patch("src.experiment_runner.JSONCache")
    @patch("src.experiment_runner.load_ground_truth")
    @patch("src.experiment_runner.load_configs")
    def test_returns_all_results(
        self,
        mock_load_configs,
        mock_load_gt,
        mock_json_cache,
        mock_create_llm,
        mock_judge_cls,
        mock_create_embedder,
        mock_run_single,
        mock_gc,
        tmp_path,
    ) -> None:
        mock_load_configs.return_value = [_bm25_config(), _bm25_config()]
        mock_load_gt.return_value = _make_ground_truth()
        mock_run_single.return_value = _make_mock_result()

        out_dir = str(tmp_path / "out")
        results = run_experiment_grid(output_dir=out_dir, run_judge=False)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# _EMBEDDER_ORDER constant
# ---------------------------------------------------------------------------


class TestEmbedderOrder:
    def test_order_is_minilm_mpnet_none_openai(self) -> None:
        assert _EMBEDDER_ORDER == ["minilm", "mpnet", None, "openai"]

    def test_none_is_in_order(self) -> None:
        assert None in _EMBEDDER_ORDER
