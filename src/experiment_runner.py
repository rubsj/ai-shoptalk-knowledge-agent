"""Experiment grid runner — runs 35+ configs.

Pools by embedding model to minimize load/unload cycles:
  1. MiniLM configs (10) → del + gc.collect()
  2. mpnet configs (10) → del + gc.collect()
  3. BM25-only configs (5, no model)
  4. OpenAI configs (API calls, zero local RAM)

Loading a SentenceTransformer takes 2-5s. Loading it 10 times = waste.
Same batch-by-resource pattern from P2/P3.

Memory: psutil.virtual_memory().percent — warns if >85%.
"""

from __future__ import annotations

import gc
import json
import logging
import tempfile
import time
import uuid
from pathlib import Path

import psutil

from src.cache import JSONCache
from src.evaluation import LLMJudge, compute_overlap_relevance, mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.evaluation.ground_truth import load_ground_truth
from src.factories import create_chunker, create_embedder, create_llm, create_reranker, create_retriever, load_configs
from src.generator import build_qa_prompt, extract_citations
from src.interfaces import BaseEmbedder
from src.schemas import (
    Document,
    ExperimentConfig,
    ExperimentResult,
    GroundTruthSet,
    JudgeResult,
    JudgeScores,
    PerformanceMetrics,
    QueryResult,
    RetrievalMetrics,
)
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

_EMBEDDER_ORDER: list[str | None] = ["minilm", "mpnet", None, "openai"]

# text-embedding-3-small pricing: $0.02 per 1M tokens
_OPENAI_EMBED_COST_PER_TOKEN = 0.02 / 1_000_000
_AVG_TOKENS_PER_CHUNK = 150  # ~512 chars ≈ 150 tokens


def _group_configs_by_embedder(
    configs: list[ExperimentConfig],
) -> dict[str | None, list[ExperimentConfig]]:
    """Group configs by embedding_model. None = BM25-only configs."""
    groups: dict[str | None, list[ExperimentConfig]] = {}
    for config in configs:
        key = config.embedding_model
        groups.setdefault(key, []).append(config)
    return groups


def _run_single_config(
    config: ExperimentConfig,
    embedder: BaseEmbedder | None,
    documents: list[Document],
    ground_truth: GroundTruthSet,
    judge: LLMJudge | None,
    cache: JSONCache | None,
) -> ExperimentResult:
    """Run one experiment config end-to-end and return ExperimentResult."""
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)
    ingestion_start = time.monotonic()

    # 1. Chunk all documents
    chunker = create_chunker(config)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    # 2. Embed and index (skip for BM25)
    vector_store: FAISSVectorStore | None = None
    if embedder is not None and config.retriever_type != "bm25":
        vector_store = FAISSVectorStore(dimension=embedder.dimensions)
        texts = [c.content for c in all_chunks]
        embeddings = embedder.embed(texts)
        vector_store.add(all_chunks, embeddings)

    ingestion_time = time.monotonic() - ingestion_start

    # 3. Create retriever
    retriever = create_retriever(
        config, embedder=embedder, chunks=all_chunks, vector_store=vector_store
    )

    # 4. Create LLM for answer generation
    llm = create_llm(cache=cache)

    # 5. Run evaluation queries with per-query judge scoring
    query_results: list[QueryResult] = []
    total_latency_ms = 0.0

    for gt_query in ground_truth.queries:
        query_start = time.monotonic()

        # Retrieve
        retrieval_results = retriever.retrieve(gt_query.question, top_k=config.top_k)

        # Apply reranking if configured
        if config.use_reranking and config.reranker_type:
            reranker = create_reranker(config.reranker_type)
            retrieval_results = reranker.rerank(gt_query.question, retrieval_results, top_k=config.top_k)

        retrieved_ids = [r.chunk.id for r in retrieval_results]
        retrieved_chunks = [r.chunk for r in retrieval_results]

        # Generate answer
        prompt = build_qa_prompt(gt_query.question, retrieved_chunks)
        answer = llm.generate(prompt)
        citations = extract_citations(answer, retrieved_chunks)

        latency_ms = (time.monotonic() - query_start) * 1000
        total_latency_ms += latency_ms

        # Compute per-query retrieval metrics via char offset overlap matching.
        # WHY: ground truth was generated with RecursiveChunker; other chunkers produce
        # different char offsets → different deterministic IDs → exact ID match = 0.0.
        # compute_overlap_relevance falls back to exact ID matching for legacy GT entries.
        relevant_ids, graded_relevance = compute_overlap_relevance(
            retrieved_chunks, gt_query.relevant_chunks
        )

        q_metrics = RetrievalMetrics(
            recall_at_5=recall_at_k(retrieved_ids, relevant_ids, k=5),
            precision_at_5=precision_at_k(retrieved_ids, relevant_ids, k=5),
            mrr=mrr(retrieved_ids, relevant_ids),
            ndcg_at_5=ndcg_at_k(retrieved_ids, graded_relevance, k=5),
        )

        # Per-query judge scoring
        judge_result: JudgeResult | None = None
        if judge is not None:
            judge_result = judge.score(gt_query.question, answer, retrieved_chunks)

        query_results.append(
            QueryResult(
                query_id=gt_query.query_id,
                question=gt_query.question,
                answer=answer,
                retrieved_chunk_ids=retrieved_ids,
                retrieval_scores=q_metrics,
                judge_result=judge_result,
                latency_ms=latency_ms,
            )
        )

    # 6. Aggregate metrics over all queries
    n = len(query_results)
    avg_metrics = RetrievalMetrics(
        recall_at_5=sum(r.retrieval_scores.recall_at_5 for r in query_results) / n,
        precision_at_5=sum(r.retrieval_scores.precision_at_5 for r in query_results) / n,
        mrr=sum(r.retrieval_scores.mrr for r in query_results) / n,
        ndcg_at_5=sum(r.retrieval_scores.ndcg_at_5 for r in query_results) / n,
    )

    # 7. Aggregate per-query judge results into JudgeScores
    judge_scores: JudgeScores | None = None
    if judge is not None:
        per_query_judges = [qr.judge_result for qr in query_results if qr.judge_result is not None]
        if per_query_judges:
            n_j = len(per_query_judges)
            avg_relevance = sum(r.relevance for r in per_query_judges) / n_j
            avg_accuracy = sum(r.accuracy for r in per_query_judges) / n_j
            avg_completeness = sum(r.completeness for r in per_query_judges) / n_j
            avg_conciseness = sum(r.conciseness for r in per_query_judges) / n_j
            avg_citation_quality = sum(r.citation_quality for r in per_query_judges) / n_j
            overall_average = (
                avg_relevance + avg_accuracy + avg_completeness + avg_conciseness + avg_citation_quality
            ) / 5
            judge_scores = JudgeScores(
                avg_relevance=avg_relevance,
                avg_accuracy=avg_accuracy,
                avg_completeness=avg_completeness,
                avg_conciseness=avg_conciseness,
                avg_citation_quality=avg_citation_quality,
                overall_average=overall_average,
            )

    # 8. Measure FAISS index size via temp save
    index_size_bytes = 0
    if vector_store is not None:
        with tempfile.NamedTemporaryFile(suffix="_idx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            vector_store.save(tmp_path)
            faiss_file = Path(f"{tmp_path}.faiss")
            if faiss_file.exists():
                index_size_bytes = faiss_file.stat().st_size
        finally:
            for ext in (".faiss", ".json"):
                p = Path(f"{tmp_path}{ext}")
                if p.exists():
                    p.unlink()

    # 9. Collect performance metrics
    mem_end_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(mem_start_mb, mem_end_mb)

    embedding_source = (
        "none"
        if config.retriever_type == "bm25"
        else "api"
        if config.embedding_model == "openai"
        else "local"
    )

    # 10. Compute cost estimate (embedding + LLM generation)
    cost_estimate_usd = 0.0
    if embedding_source == "api":
        total_tokens = len(all_chunks) * _AVG_TOKENS_PER_CHUNK
        cost_estimate_usd += total_tokens * _OPENAI_EMBED_COST_PER_TOKEN
    # LLM generation cost: gpt-4o-mini at ~$0.15/1M input + ~$0.60/1M output
    # Rough estimate: ~1500 input tokens per query (prompt + chunks), ~200 output tokens
    n_queries = len(ground_truth.queries)
    cost_estimate_usd += n_queries * (1500 * 0.15 + 200 * 0.60) / 1_000_000

    performance = PerformanceMetrics(
        ingestion_time_seconds=ingestion_time,
        avg_query_latency_ms=total_latency_ms / n if n > 0 else 0.0,
        index_size_bytes=index_size_bytes,
        peak_memory_mb=peak_memory_mb,
        embedding_source=embedding_source,
        cost_estimate_usd=cost_estimate_usd,
    )

    return ExperimentResult(
        experiment_id=str(uuid.uuid4()),
        config=config,
        metrics=avg_metrics,
        judge_scores=judge_scores,
        performance=performance,
        query_results=query_results,
    )


def run_experiment_grid(
    config_dir: str = "experiments/configs",
    ground_truth_path: str = "data/ground_truth.json",
    output_dir: str = "results/experiments",
    documents: list[Document] | None = None,
    run_judge: bool = True,
) -> list[ExperimentResult]:
    """Run the full experiment grid, pooling embedder loading across configs.

    Groups configs by embedding_model so each model is loaded exactly once.
    Execution order follows _EMBEDDER_ORDER: minilm → mpnet → None → openai.
    Between groups: del embedder + gc.collect() to free RAM.

    Saves per-experiment JSON to {output_dir}/{experiment_id}.json and
    a summary.json with all results for easy loading.
    """
    if documents is None:
        documents = []

    configs = load_configs(config_dir)
    ground_truth = load_ground_truth(ground_truth_path)

    cache_dir = str(Path(output_dir) / "llm_cache")
    cache = JSONCache(cache_dir)
    judge: LLMJudge | None = None
    if run_judge:
        judge = LLMJudge(model="gpt-4o", cache=cache)

    groups = _group_configs_by_embedder(configs)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results: list[ExperimentResult] = []
    for embedder_name in _EMBEDDER_ORDER:
        if embedder_name not in groups:
            continue

        group_configs = groups[embedder_name]
        mem_pct = psutil.virtual_memory().percent
        if mem_pct > 85:
            logger.warning("Memory usage %.1f%% — above 85%% threshold before loading embedder", mem_pct)

        embedder: BaseEmbedder | None = (
            create_embedder(embedder_name) if embedder_name is not None else None
        )

        for config in group_configs:
            logger.info(
                "Running: chunker=%s  embedder=%s  retriever=%s",
                config.chunking_strategy,
                config.embedding_model,
                config.retriever_type,
            )
            result = _run_single_config(
                config=config,
                embedder=embedder,
                documents=documents,
                ground_truth=ground_truth,
                judge=judge,
                cache=cache,
            )
            all_results.append(result)

            # Save per-experiment JSON incrementally
            result_file = out_path / f"{result.experiment_id}.json"
            result_file.write_text(json.dumps(result.model_dump(mode="json"), indent=2, default=str))

        del embedder
        gc.collect()

    # Save summary with all results
    summary_data = [r.model_dump(mode="json") for r in all_results]
    (out_path / "summary.json").write_text(json.dumps(summary_data, indent=2, default=str))
    logger.info("Saved %d results to %s", len(all_results), output_dir)

    return all_results
