"""Ground truth management — loading, validation, and generation helpers.

Ground truth schema:
    queries: list of {query_id, question, relevant_chunks: [{chunk_id, relevance_grade}]}
    relevance_grade: int 0-3 (PRD Decision 3 scale)

Generation workflow:
    1. LLM generates 30 candidate QA pairs with chunk mappings
    2. Developer curates 15, assigning relevance grades manually
    3. Result saved to data/ground_truth.json

Hybrid approach avoids circular evaluation pitfall: LLM-generated questions
tested against the same LLM's retrieval would inflate quality metrics.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import instructor
import litellm

from src.schemas import Chunk, GeneratedQAPair, GroundTruthSet

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10

_SYSTEM_PROMPT = """You are a retrieval evaluation expert. Given a batch of document chunks, \
generate one question that is directly answerable by 1-3 chunks in the batch.

For every chunk in the batch, assign a relevance grade:
  3 = Gold: directly answers the question
  2 = Same section: contextually relevant but not the primary answer
  1 = Same document: topically related but peripheral
  0 = Irrelevant: no connection to the question

Only include chunks with grade >= 1 in relevant_chunks.
Return a single JSON object matching the GeneratedQAPair schema."""


def load_ground_truth(path: str) -> GroundTruthSet:
    """Read JSON from *path*, validate through GroundTruthSet. Pydantic catches bad data."""
    raw = Path(path).read_text(encoding="utf-8")
    import json

    data = json.loads(raw)
    return GroundTruthSet.model_validate(data)


def generate_ground_truth_candidates(
    chunks: list[Chunk],
    n: int = 30,
    model: str = "gpt-4o",
) -> list[GeneratedQAPair]:
    """LLM generates up to *n* QA pairs from *chunks* via Instructor.

    Chunks are batched in groups of ~10.  Each batch produces one GeneratedQAPair.
    The prompt includes chunk IDs and content so the LLM can reference them by ID.
    Instructor handles JSON parsing, Pydantic validation, and automatic retry (max 3).

    Returns validated GeneratedQAPair models for human curation.
    """
    if not chunks:
        logger.warning("generate_ground_truth_candidates called with empty chunk list")
        return []

    client = instructor.from_litellm(litellm.completion)

    # How many batches do we need to reach n candidates?
    n_batches = math.ceil(n / 1)  # one pair per batch
    batches: list[list[Chunk]] = []
    for i in range(0, len(chunks), _BATCH_SIZE):
        batches.append(chunks[i : i + _BATCH_SIZE])
        if len(batches) >= n_batches:
            break

    results: list[GeneratedQAPair] = []
    for batch_idx, batch in enumerate(batches):
        chunk_block = "\n\n".join(
            f'[{chunk.id}]: "{chunk.content}"' for chunk in batch
        )
        user_prompt = (
            f"Here are {len(batch)} document chunks:\n\n"
            f"{chunk_block}\n\n"
            "Generate one question answerable by 1-3 of these chunks and assign relevance "
            "grades (0-3) to each chunk. Only include chunks with grade >= 1 in relevant_chunks."
        )
        try:
            pair: GeneratedQAPair = client.chat.completions.create(
                model=model,
                response_model=GeneratedQAPair,
                max_retries=3,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            results.append(pair)
            logger.debug("Batch %d/%d: generated question %r", batch_idx + 1, len(batches), pair.question[:60])
        except Exception:
            logger.exception("Batch %d/%d: generation failed, skipping", batch_idx + 1, len(batches))

    logger.info("Generated %d/%d QA pairs", len(results), len(batches))
    return results
