"""Generate ground truth candidates for RAG evaluation — DEVELOPER CURATION GATE.

Workflow:
    1. Load extracted documents from disk cache
    2. Chunk with RecursiveChunker(512, 50) — baseline chunker
    3. Call GPT-4o to generate 30 QA candidates (one per chunk batch)
    4. Enrich candidates with document_id, start_char, end_char, text_preview
    5. Save to data/ground_truth_candidates.json for developer curation
    6. Developer curates → saves as data/ground_truth.json

Usage:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --n 20 --model gpt-4o-mini
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunkers.recursive import RecursiveChunker  # noqa: E402
from src.evaluation.ground_truth import generate_ground_truth_candidates  # noqa: E402
from src.extraction import extract_all_pdfs  # noqa: E402
from src.schemas import Chunk, GeneratedQAPair  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _chunk_id_to_chunk(chunks: list[Chunk]) -> dict[str, Chunk]:
    """Build a lookup from chunk ID to Chunk object."""
    return {c.id: c for c in chunks}


def _format_candidate(
    idx: int,
    pair: GeneratedQAPair,
    chunk_lookup: dict[str, Chunk],
) -> dict:
    """Format a GeneratedQAPair into a developer-friendly dict for curation."""
    relevant_sections = []
    for gt_chunk in pair.relevant_chunks:
        chunk = chunk_lookup.get(gt_chunk.chunk_id)
        if chunk is None:
            logger.warning("Chunk ID %s not found in lookup, skipping", gt_chunk.chunk_id)
            continue
        relevant_sections.append({
            "chunk_id": gt_chunk.chunk_id,
            "document_id": chunk.metadata.document_id,
            "source": chunk.metadata.source,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "page_number": chunk.metadata.page_number,
            "text_preview": chunk.content[:150],
            "suggested_grade": gt_chunk.relevance_grade,
        })

    return {
        "query_id": f"q{idx + 1:02d}",
        "question": pair.question,
        "relevant_sections": relevant_sections,
    }


def _format_ground_truth_entry(candidate: dict) -> dict:
    """Convert a candidate dict into GroundTruthSet-compatible format."""
    return {
        "query_id": candidate["query_id"],
        "question": candidate["question"],
        "relevant_chunks": [
            {
                "chunk_id": sec["chunk_id"],
                "document_id": sec["document_id"],
                "start_char": sec["start_char"],
                "end_char": sec["end_char"],
                "relevance_grade": sec["suggested_grade"],
            }
            for sec in candidate["relevant_sections"]
        ],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth candidates")
    parser.add_argument("--n", type=int, default=30, help="Number of QA pairs to generate")
    parser.add_argument("--model", default="gpt-4o", help="LLM model for generation")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="PDF directory")
    parser.add_argument("--output", default="data/ground_truth_candidates.json", help="Output path")
    args = parser.parse_args()

    # 1. Load documents from cache
    print("Loading documents from cache...")
    documents = extract_all_pdfs(args.pdf_dir)
    print(f"  {len(documents)} documents, {sum(len(d.content) for d in documents):,} chars total")

    # 2. Chunk with RecursiveChunker(512, 50)
    print("Chunking with RecursiveChunker(512, 50)...")
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunker.chunk(doc)
        all_chunks.extend(doc_chunks)
        print(f"  {Path(doc.metadata.source).name}: {len(doc_chunks)} chunks")
    print(f"  Total: {len(all_chunks)} chunks")

    chunk_lookup = _chunk_id_to_chunk(all_chunks)

    # 3. Generate candidates via GPT-4o
    print(f"\nGenerating {args.n} QA candidates with {args.model}...")
    print("  (This calls the API — ~$0.10-0.30 for 30 candidates with gpt-4o)")
    pairs = generate_ground_truth_candidates(all_chunks, n=args.n, model=args.model)
    print(f"  Generated {len(pairs)} candidates")

    if not pairs:
        print("ERROR: No candidates generated. Check API key and model.", file=sys.stderr)
        return 1

    # 4. Enrich with metadata and save candidates
    candidates = [
        _format_candidate(i, pair, chunk_lookup)
        for i, pair in enumerate(pairs)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(candidates)} candidates to {output_path}")

    # 5. Also generate a draft ground_truth.json for convenience
    draft_gt = {
        "queries": [_format_ground_truth_entry(c) for c in candidates]
    }
    draft_path = Path("data/ground_truth_draft.json")
    draft_path.write_text(
        json.dumps(draft_gt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved draft ground truth to {draft_path}")

    # Print per-paper distribution
    paper_counts: dict[str, int] = {}
    for c in candidates:
        for sec in c["relevant_sections"]:
            stem = Path(sec["source"]).stem
            paper_counts[stem] = paper_counts.get(stem, 0) + 1
    print("\nPaper distribution:")
    for paper, count in sorted(paper_counts.items()):
        print(f"  {paper}: {count} relevant sections")

    # 6. Print curation instructions
    print(f"""
{'=' * 72}
  GROUND TRUTH CURATION — DEVELOPER REQUIRED
{'=' * 72}

  {len(candidates)} LLM-generated candidates saved to {output_path}
  Draft ground truth saved to {draft_path}

  Open {output_path} and perform these steps:

  a) DELETE questions answerable from general knowledge (not PDF-specific)
  b) DELETE questions where the answer is trivially in one obvious chunk
     (too easy — won't discriminate between retrieval configs)
  c) DELETE duplicate or near-duplicate questions
  d) VERIFY remaining queries span all 4 papers (minimum 2 per paper)
  e) VERIFY difficulty mix: ~5 easy (1 gold chunk), ~5 medium (2 chunks),
     ~5 hard (synthesis across sections)
  f) For each kept query, review the relevant_sections and assign grades:
     3 = directly answers | 2 = same section | 1 = same document | 0 = irrelevant
     The text_preview field shows what each section contains.
     You may adjust start_char/end_char boundaries.
  g) Target: 15 queries minimum

  After curation, save as data/ground_truth.json matching GroundTruthSet schema.
  You can start from {draft_path} and remove unwanted queries.

  Validate:
    python -c "from src.evaluation.ground_truth import load_ground_truth; \\
      gt = load_ground_truth('data/ground_truth.json'); \\
      print(f'{{len(gt.queries)}} queries loaded, valid')"

{'=' * 72}
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
