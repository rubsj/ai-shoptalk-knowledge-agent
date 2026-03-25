"""5-axis LLM-as-Judge for generation quality evaluation.

Axes (1–5 scale each):
    Relevance       — answer addresses the question
    Accuracy        — every claim verifiable in provided context
    Completeness    — answer is thorough
    Conciseness     — not unnecessarily verbose
    Citation Quality — proper [N] source attribution

Uses Instructor for structured output (same pattern as P1/P4 judge).
Default model: gpt-4o (evaluation requires stronger reasoning than generation).

Why not RAGAS: P2 already used RAGAS. Custom judge gives per-axis diagnostics
and demonstrates understanding of what RAGAS does internally.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm

from src.cache import JSONCache
from src.schemas import Chunk, JudgeResult, JudgeScores

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are an expert RAG evaluation judge. \
Score the answer on 5 axes using a 1-5 integer scale.

Rubric:
  Relevance:        1=off-topic, 3=partially addresses, 5=directly answers the question
  Accuracy:         1=major errors, 3=minor errors, 5=every claim verifiable in provided context
  Completeness:     1=fragment, 3=covers basics, 5=comprehensive and thorough
  Conciseness:      1=extremely verbose, 3=some filler, 5=focused and appropriately brief
  Citation Quality: 1=no/wrong citations, 3=some correct [N] markers, 5=every claim properly cited

Score strictly — a 5 means near-perfect on that axis."""


class LLMJudge:
    """5-axis LLM-as-Judge using Instructor for structured output."""

    def __init__(self, model: str = "gpt-4o", cache: JSONCache | None = None) -> None:
        self._model = model
        self._cache = cache
        self._client = instructor.from_litellm(litellm.completion)

    def score(self, query: str, answer: str, chunks: list[Chunk]) -> JudgeResult:
        """Score one answer on 5 axes. Checks cache first; stores on miss.

        Returns a validated JudgeResult. Instructor handles retry (max 3).
        """
        chunk_context = "\n\n".join(
            f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(chunks)
        )
        user_prompt = (
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Source chunks:\n{chunk_context}"
        )

        if self._cache is not None:
            key = self._cache.make_key(self._model, _JUDGE_SYSTEM_PROMPT, user_prompt)
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for judge key %s", key)
                return JudgeResult.model_validate(cached)

        result: JudgeResult = self._client.chat.completions.create(
            model=self._model,
            response_model=JudgeResult,
            max_retries=3,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        if self._cache is not None:
            self._cache.set(key, result.model_dump())

        return result

    def score_batch(self, qa_pairs: list[dict[str, Any]]) -> JudgeScores:
        """Score all pairs and aggregate per-axis averages.

        Each item in qa_pairs must have keys: 'query', 'answer', 'chunks'.
        """
        judge_results: list[JudgeResult] = []
        for pair in qa_pairs:
            result = self.score(
                query=pair["query"],
                answer=pair["answer"],
                chunks=pair.get("chunks", []),
            )
            judge_results.append(result)

        n = len(judge_results)
        avg_relevance = sum(r.relevance for r in judge_results) / n
        avg_accuracy = sum(r.accuracy for r in judge_results) / n
        avg_completeness = sum(r.completeness for r in judge_results) / n
        avg_conciseness = sum(r.conciseness for r in judge_results) / n
        avg_citation_quality = sum(r.citation_quality for r in judge_results) / n
        overall_average = (
            avg_relevance
            + avg_accuracy
            + avg_completeness
            + avg_conciseness
            + avg_citation_quality
        ) / 5

        return JudgeScores(
            avg_relevance=avg_relevance,
            avg_accuracy=avg_accuracy,
            avg_completeness=avg_completeness,
            avg_conciseness=avg_conciseness,
            avg_citation_quality=avg_citation_quality,
            overall_average=overall_average,
        )
