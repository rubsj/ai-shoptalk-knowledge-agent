"""LLM generation with citation extraction via LiteLLM.

LiteLLMClient wraps any provider (OpenAI, Anthropic, Cohere) behind one
interface. Default: gpt-4o-mini for generation, gpt-4o for evaluation.

Citation extraction: regex parses [N] markers from answers, validates N is
within chunk range, maps to Citation objects. Parse-only — LLM-as-Judge
handles semantic citation quality.

LiteLLM over raw OpenAI SDK: multi-provider support per spec. Minimal code
difference, but lets us swap providers without touching call sites.
"""

from __future__ import annotations

import logging
import re

import litellm

from src.cache import JSONCache
from src.interfaces import BaseLLM
from src.schemas import Citation, Chunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class LiteLLMClient(BaseLLM):
    """LiteLLM-backed LLM client with optional JSON file cache."""

    def __init__(
        self, model: str = "gpt-4o-mini", cache: JSONCache | None = None
    ) -> None:
        self._model = model
        self._cache = cache

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion, using cache when available."""
        if self._cache is not None:
            key = self._cache.make_key(self._model, system_prompt, prompt)
            cached = self._cache.get(key)
            if cached is not None:
                return cached["content"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = litellm.completion(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        content: str = response.choices[0].message.content

        if self._cache is not None:
            self._cache.set(key, {"content": content})

        return content


def build_qa_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build a numbered-context prompt instructing the LLM to cite with [N]."""
    context_lines = [
        f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(chunks)
    ]
    context_block = "\n\n".join(context_lines)
    return (
        f"Answer the following question using only the provided context. "
        f"Cite sources with [N] markers where N corresponds to the context number.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )


def extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Parse [N] markers from answer and map to Citation objects.

    Out-of-range markers are warned and skipped. Duplicates are deduplicated
    (first occurrence wins). Citation relevance_score is set to 0.0 — LLM-as-Judge
    sets the real score on Day 3.
    """
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n < 1 or n > len(chunks):
            logger.warning("Citation [%d] out of range (1–%d), skipping.", n, len(chunks))
            continue
        if n in seen:
            continue
        seen.add(n)
        chunk = chunks[n - 1]
        citations.append(
            Citation(
                chunk_id=chunk.id,
                source=chunk.metadata.source,
                page_number=chunk.metadata.page_number,
                text_snippet=chunk.content[:100],
                relevance_score=0.0,
            )
        )

    return citations