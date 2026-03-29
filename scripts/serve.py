"""CLI: interactive QA REPL — loads index, accepts questions, prints cited answers.

Usage:
    python scripts/serve.py --config experiments/configs/01_fixed_minilm_dense.yaml
    python scripts/serve.py --config experiments/configs/01_fixed_minilm_dense.yaml --model gpt-4o

Deliverable D3 (PRD Section 7a): question → cited answer in < 5 seconds.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedders.ollama_embedder import OllamaUnavailableError  # noqa: E402
from src.factories import create_embedder, create_llm, create_reranker, create_retriever  # noqa: E402
from src.generator import build_qa_prompt, extract_citations  # noqa: E402
from src.schemas import ExperimentConfig  # noqa: E402
from src.vector_store import FAISSVectorStore  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML experiment config path",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="LLM model for answer generation (default: gpt-4o-mini)",
)
def serve(config: Path, model: str) -> None:
    """Load a FAISS index and start an interactive QA REPL."""
    # Load and validate config
    raw = yaml.safe_load(config.read_text())
    try:
        cfg = ExperimentConfig.model_validate(raw)
    except Exception as e:
        click.echo(f"ERROR: Invalid config {config}: {e}", err=True)
        raise SystemExit(1)

    if cfg.embedding_model is None:
        click.echo(
            "ERROR: BM25-only configs have no FAISS index to serve. "
            "Use a config with an embedding_model.",
            err=True,
        )
        raise SystemExit(1)

    # Derive index path from config filename stem (matches ingest.py convention)
    index_dir = Path("data/indices") / config.stem
    index_path = str(index_dir / "index")

    # Validate index files exist
    if not Path(f"{index_path}.faiss").exists() or not Path(f"{index_path}.json").exists():
        click.echo(
            f"ERROR: Index not found at {index_dir}/. "
            f"Run ingest first:\n  python scripts/ingest.py --config {config}",
            err=True,
        )
        raise SystemExit(1)

    # Build pipeline components
    click.echo(f"Loading embedder ({cfg.embedding_model})...")
    try:
        embedder = create_embedder(cfg.embedding_model)
    except OllamaUnavailableError:
        click.echo(
            "ERROR: Ollama is not running. Start it with:\n"
            "  ollama serve\n"
            "  ollama pull nomic-embed-text",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Loading index from {index_dir}/...")
    store = FAISSVectorStore(dimension=embedder.dimensions)
    store.load(index_path)

    # create_retriever needs chunks for BM25 sub-retriever in hybrid mode
    retriever = create_retriever(cfg, embedder, store.chunks, store)
    reranker = create_reranker(cfg.reranker_type) if cfg.use_reranking else None
    llm = create_llm(model=model)

    rerank_label = f" + rerank({cfg.reranker_type})" if cfg.use_reranking else ""
    click.echo(
        f"\nReady — {cfg.chunking_strategy} / {cfg.embedding_model} / "
        f"{cfg.retriever_type}{rerank_label} / top_k={cfg.top_k}"
    )
    click.echo("Type a question, or 'exit' / 'quit' to stop.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            click.echo("Exiting.")
            break

        start = time.monotonic()

        # Retrieve
        results = retriever.retrieve(query, top_k=cfg.top_k)

        # Optional rerank
        if reranker is not None:
            results = reranker.rerank(query, results, top_k=cfg.top_k)

        context_chunks = [r.chunk for r in results]

        # Generate
        prompt = build_qa_prompt(query, context_chunks)
        answer = llm.generate(prompt)

        # Extract citations
        citations = extract_citations(answer, context_chunks)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Display answer
        click.echo(f"\nAnswer ({elapsed_ms:.0f}ms):\n{answer}")

        # Display sources
        if citations:
            click.echo("\nSources:")
            for i, cit in enumerate(citations, 1):
                source_name = Path(cit.source).name
                page = cit.page_number + 1  # 0-indexed internally
                click.echo(f"  [{i}] {source_name} — page {page}")
                click.echo(f"      {cit.text_snippet}...")
        click.echo()


if __name__ == "__main__":
    serve()
