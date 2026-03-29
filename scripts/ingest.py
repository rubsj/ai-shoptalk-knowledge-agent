"""CLI: PDF ingestion pipeline — PDF → chunk → embed → FAISS index → save to disk.

Usage:
    python scripts/ingest.py --config experiments/configs/01_fixed_minilm_dense.yaml
    python scripts/ingest.py --config experiments/configs/01_fixed_minilm_dense.yaml --pdf-dir data/pdfs/

Deliverable D1 (PRD Section 7a): verifiable by index files appearing on disk.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedders.ollama_embedder import OllamaUnavailableError  # noqa: E402
from src.extraction import extract_all_pdfs  # noqa: E402
from src.factories import create_chunker, create_embedder  # noqa: E402
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
    "--pdf-dir",
    default="data/pdfs",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing PDF files (default: data/pdfs)",
)
def ingest(config: Path, pdf_dir: Path) -> None:
    """Ingest PDFs into a FAISS index using the given experiment config."""
    # Load and validate config
    raw = yaml.safe_load(config.read_text())
    try:
        cfg = ExperimentConfig.model_validate(raw)
    except Exception as e:
        click.echo(f"ERROR: Invalid config {config}: {e}", err=True)
        raise SystemExit(1)

    # BM25-only configs have no embedding model — cannot produce a FAISS index
    if cfg.embedding_model is None:
        click.echo(
            "ERROR: BM25-only configs do not produce a FAISS index. "
            "Use evaluate.py for BM25 experiments.",
            err=True,
        )
        raise SystemExit(1)

    # Derive output path from config filename stem
    index_dir = Path("data/indices") / config.stem
    index_path = str(index_dir / "index")

    click.echo(f"Config:    {config}")
    click.echo(f"PDF dir:   {pdf_dir}")
    click.echo(f"Index out: {index_dir}/")

    start_total = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:

        # --- Extract ---
        task = progress.add_task("Extracting PDFs...", total=None)
        t0 = time.monotonic()
        documents = extract_all_pdfs(str(pdf_dir))
        progress.update(task, description=f"Extracted {len(documents)} document(s)")
        extract_time = time.monotonic() - t0

        # --- Chunk ---
        progress.update(task, description="Chunking documents...")
        t0 = time.monotonic()
        chunker = create_chunker(cfg)
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk(doc))
        progress.update(task, description=f"Created {len(all_chunks)} chunks")
        chunk_time = time.monotonic() - t0

        # --- Embed ---
        progress.update(task, description=f"Embedding with {cfg.embedding_model}...")
        t0 = time.monotonic()
        try:
            embedder = create_embedder(cfg.embedding_model)
        except OllamaUnavailableError:
            click.echo(
                "\nERROR: Ollama is not running. Start it with:\n"
                "  ollama serve\n"
                "  ollama pull nomic-embed-text",
                err=True,
            )
            raise SystemExit(1)
        embeddings = embedder.embed([c.content for c in all_chunks])
        embed_time = time.monotonic() - t0

        # --- Index ---
        progress.update(task, description="Building FAISS index...")
        t0 = time.monotonic()
        index_dir.mkdir(parents=True, exist_ok=True)
        store = FAISSVectorStore(dimension=embedder.dimensions)
        store.add(all_chunks, embeddings)
        store.save(index_path)
        index_time = time.monotonic() - t0

    total_time = time.monotonic() - start_total

    # Summary
    faiss_size = Path(f"{index_path}.faiss").stat().st_size / 1024
    json_size = Path(f"{index_path}.json").stat().st_size / 1024

    click.echo(f"\nDone in {total_time:.1f}s")
    click.echo(f"  Documents:  {len(documents)}")
    click.echo(f"  Chunks:     {len(all_chunks)}")
    click.echo(f"  Dimensions: {embedder.dimensions}")
    click.echo(f"  Index size: {faiss_size:.1f} KB (.faiss) + {json_size:.1f} KB (.json)")
    click.echo(f"  Timings:    extract={extract_time:.1f}s  chunk={chunk_time:.1f}s  "
               f"embed={embed_time:.1f}s  index={index_time:.1f}s")
    click.echo(f"  Saved to:   {index_dir}/")


if __name__ == "__main__":
    ingest()
