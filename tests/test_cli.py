"""Tests for Click CLI scripts: ingest.py, serve.py, evaluate.py.

Uses Click's CliRunner — no subprocesses, no API keys, no Ollama required.
All external dependencies are mocked at module import boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

# Add project root so scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config_yaml(
    chunking_strategy: str = "fixed",
    embedding_model: str | None = "minilm",
    retriever_type: str = "dense",
) -> dict:
    cfg: dict = {
        "chunking_strategy": chunking_strategy,
        "embedding_model": embedding_model,
        "retriever_type": retriever_type,
        "top_k": 5,
    }
    if retriever_type == "hybrid":
        cfg["hybrid_alpha"] = 0.7
    if embedding_model is None:
        cfg.pop("embedding_model")
        cfg["embedding_model"] = None
    return cfg


# ---------------------------------------------------------------------------
# ingest.py
# ---------------------------------------------------------------------------


class TestIngestCli:
    def test_help_shows_options(self):
        from scripts.ingest import ingest

        result = CliRunner().invoke(ingest, ["--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--pdf-dir" in result.output

    def test_missing_config_fails(self, tmp_path):
        from scripts.ingest import ingest

        result = CliRunner().invoke(ingest, ["--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0

    def test_bm25_only_config_exits_with_error(self, tmp_path):
        from scripts.ingest import ingest

        cfg = _make_config_yaml(embedding_model=None, retriever_type="bm25")
        config_file = tmp_path / "bm25.yaml"
        config_file.write_text(yaml.dump(cfg))

        result = CliRunner().invoke(ingest, ["--config", str(config_file), "--pdf-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "BM25" in result.output

    def test_ollama_unavailable_exits_with_message(self, tmp_path):
        from scripts.ingest import ingest
        from src.embedders.ollama_embedder import OllamaUnavailableError

        cfg = _make_config_yaml(embedding_model="ollama_nomic")
        config_file = tmp_path / "ollama.yaml"
        config_file.write_text(yaml.dump(cfg))

        with patch("scripts.ingest.create_embedder", side_effect=OllamaUnavailableError("no ollama")):
            with patch("scripts.ingest.extract_all_pdfs", return_value=[]):
                result = CliRunner().invoke(
                    ingest,
                    ["--config", str(config_file), "--pdf-dir", str(tmp_path)],
                )
        assert result.exit_code == 1
        assert "Ollama" in result.output

    def test_ingest_creates_index_files(self, tmp_path):
        from scripts.ingest import ingest
        from src.schemas import Chunk, ChunkMetadata, Document, DocumentMetadata, PageInfo

        cfg = _make_config_yaml()
        config_file = tmp_path / "01_test.yaml"
        config_file.write_text(yaml.dump(cfg))

        doc = Document(
            content="hello world",
            metadata=DocumentMetadata(source="test.pdf", page_count=1),
            pages=[PageInfo(page_number=0, text="hello world", char_count=11)],
        )
        chunk = Chunk(
            content="hello world",
            metadata=ChunkMetadata(
                document_id=doc.id,
                source="test.pdf",
                page_number=0,
                start_char=0,
                end_char=11,
                chunk_index=0,
            ),
        )

        mock_embedder = MagicMock()
        mock_embedder.dimensions = 4
        mock_embedder.embed.return_value = np.random.rand(1, 4).astype(np.float32)

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]

        mock_store = MagicMock()

        with patch("scripts.ingest.extract_all_pdfs", return_value=[doc]):
            with patch("scripts.ingest.create_chunker", return_value=mock_chunker):
                with patch("scripts.ingest.create_embedder", return_value=mock_embedder):
                    with patch("scripts.ingest.FAISSVectorStore", return_value=mock_store):
                        with patch("scripts.ingest.Path") as mock_path_cls:
                            # Allow real Path operations for config loading but mock index dir
                            real_path = Path
                            def path_side_effect(*args):
                                result = real_path(*args)
                                return result
                            mock_path_cls.side_effect = path_side_effect

                            result = CliRunner().invoke(
                                ingest,
                                ["--config", str(config_file), "--pdf-dir", str(tmp_path)],
                            )

        # The mocked store.save() should have been called
        assert mock_store.add.called
        assert mock_store.save.called


# ---------------------------------------------------------------------------
# serve.py
# ---------------------------------------------------------------------------


class TestServeCli:
    def test_help_shows_options(self):
        from scripts.serve import serve

        result = CliRunner().invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output

    def test_missing_config_fails(self, tmp_path):
        from scripts.serve import serve

        result = CliRunner().invoke(serve, ["--config", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0

    def test_bm25_only_config_exits_with_error(self, tmp_path):
        from scripts.serve import serve

        cfg = _make_config_yaml(embedding_model=None, retriever_type="bm25")
        config_file = tmp_path / "bm25.yaml"
        config_file.write_text(yaml.dump(cfg))

        result = CliRunner().invoke(serve, ["--config", str(config_file)])
        assert result.exit_code == 1
        assert "BM25" in result.output

    def test_missing_index_exits_with_error(self, tmp_path):
        from scripts.serve import serve

        cfg = _make_config_yaml()
        config_file = tmp_path / "01_test.yaml"
        config_file.write_text(yaml.dump(cfg))

        result = CliRunner().invoke(serve, ["--config", str(config_file)])
        assert result.exit_code == 1
        assert "Index not found" in result.output

    def test_ollama_unavailable_exits_with_message(self, tmp_path):
        from scripts.serve import serve
        from src.embedders.ollama_embedder import OllamaUnavailableError

        cfg = _make_config_yaml(embedding_model="ollama_nomic")
        config_file = tmp_path / "01_ollama.yaml"
        config_file.write_text(yaml.dump(cfg))

        # Pretend index files exist
        index_dir = Path("data/indices") / config_file.stem
        index_faiss = Path(f"{index_dir}/index.faiss")
        index_json = Path(f"{index_dir}/index.json")

        with patch.object(Path, "exists", return_value=True):
            with patch("scripts.serve.create_embedder", side_effect=OllamaUnavailableError("no ollama")):
                result = CliRunner().invoke(serve, ["--config", str(config_file)])

        assert result.exit_code == 1
        assert "Ollama" in result.output

    def test_serve_repl_exits_on_quit(self, tmp_path):
        from scripts.serve import serve

        cfg = _make_config_yaml()
        config_file = tmp_path / "01_test.yaml"
        config_file.write_text(yaml.dump(cfg))

        mock_embedder = MagicMock()
        mock_embedder.dimensions = 4

        mock_store = MagicMock()
        mock_store.chunks = []

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Test answer."

        with patch.object(Path, "exists", return_value=True):
            with patch("scripts.serve.create_embedder", return_value=mock_embedder):
                with patch("scripts.serve.FAISSVectorStore", return_value=mock_store):
                    with patch("scripts.serve.create_retriever", return_value=mock_retriever):
                        with patch("scripts.serve.create_llm", return_value=mock_llm):
                            with patch("scripts.serve.build_qa_prompt", return_value="prompt"):
                                with patch("scripts.serve.extract_citations", return_value=[]):
                                    result = CliRunner().invoke(
                                        serve,
                                        ["--config", str(config_file)],
                                        input="quit\n",
                                    )

        assert result.exit_code == 0
        assert "Exiting" in result.output

    def test_serve_repl_answers_question(self, tmp_path):
        from scripts.serve import serve
        from src.schemas import Citation

        cfg = _make_config_yaml()
        config_file = tmp_path / "01_test.yaml"
        config_file.write_text(yaml.dump(cfg))

        mock_embedder = MagicMock()
        mock_embedder.dimensions = 4

        mock_store = MagicMock()
        mock_store.chunks = []

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The answer is 42."

        with patch.object(Path, "exists", return_value=True):
            with patch("scripts.serve.create_embedder", return_value=mock_embedder):
                with patch("scripts.serve.FAISSVectorStore", return_value=mock_store):
                    with patch("scripts.serve.create_retriever", return_value=mock_retriever):
                        with patch("scripts.serve.create_llm", return_value=mock_llm):
                            with patch("scripts.serve.build_qa_prompt", return_value="prompt"):
                                with patch("scripts.serve.extract_citations", return_value=[]):
                                    result = CliRunner().invoke(
                                        serve,
                                        ["--config", str(config_file)],
                                        input="What is the answer?\nquit\n",
                                    )

        assert result.exit_code == 0
        assert "The answer is 42." in result.output


# ---------------------------------------------------------------------------
# evaluate.py
# ---------------------------------------------------------------------------


class TestEvaluateCli:
    def test_help_shows_all_options(self):
        from scripts.evaluate import main

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        for opt in ["--configs", "--ground-truth", "--output", "--pdfs", "--no-judge", "--reproducibility-check"]:
            assert opt in result.output

    def test_missing_configs_dir_fails(self, tmp_path):
        from scripts.evaluate import main

        result = CliRunner().invoke(
            main,
            [
                "--configs",
                str(tmp_path / "nonexistent"),
                "--ground-truth",
                str(tmp_path / "gt.json"),
                "--pdfs",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0

    def test_no_judge_flag_accepted(self, tmp_path):
        from scripts.evaluate import main
        from src.schemas import (
            ExperimentConfig, ExperimentResult, JudgeScores, PerformanceMetrics, RetrievalMetrics
        )

        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        gt_file = tmp_path / "gt.json"
        gt_file.write_text('{"queries": []}')

        mock_result = MagicMock()
        mock_result.config = ExperimentConfig(
            chunking_strategy="fixed",
            embedding_model="minilm",
            retriever_type="dense",
        )
        mock_result.metrics = RetrievalMetrics(
            recall_at_5=0.8, precision_at_5=0.4, mrr=0.7, ndcg_at_5=0.75
        )
        mock_result.performance = PerformanceMetrics(
            ingestion_time_seconds=1.0,
            avg_query_latency_ms=100.0,
            index_size_bytes=1024,
            peak_memory_mb=50.0,
            embedding_source="local",
            cost_estimate_usd=0.0,
        )
        mock_result.judge_scores = None

        with patch("scripts.evaluate.extract_all_pdfs", return_value=[]):
            with patch("scripts.evaluate.run_experiment_grid", return_value=[mock_result]):
                result = CliRunner().invoke(
                    main,
                    [
                        "--configs", str(configs_dir),
                        "--ground-truth", str(gt_file),
                        "--output", str(tmp_path / "results"),
                        "--pdfs", str(tmp_path),
                        "--no-judge",
                    ],
                )

        assert result.exit_code == 0
        assert "EXPERIMENT GRID RESULTS" in result.output
