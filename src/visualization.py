"""Experiment result visualization — 10 charts answering the 4 required questions.

Charts:
  1.  Config × Metric heatmap (overview of all configs vs all retrieval metrics)
  2.  Chunking strategy comparison (answers Q1)
  3.  Embedding model comparison (answers Q4)
  4.  Dense vs BM25 vs Hybrid (answers Q2)
  5.  Hybrid alpha sweep (optimal α identification)
  6.  Reranking before/after (answers Q3)
  7.  NDCG@5 distribution per config family
  8.  LLM Judge 5-axis radar (generation quality for top configs)
  9.  Latency vs quality scatter (performance trade-off)
  10. Per-query difficulty analysis (where the system struggles)

All charts saved as PNGs to results/charts/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.schemas import ExperimentResult

matplotlib.use("Agg")

_METRICS = ["recall_at_5", "precision_at_5", "mrr", "ndcg_at_5"]
_METRIC_LABELS = {"recall_at_5": "Recall@5", "precision_at_5": "Precision@5",
                   "mrr": "MRR", "ndcg_at_5": "NDCG@5"}
_STYLE = "seaborn-v0_8-whitegrid"
_FIGSIZE = (12, 8)
_DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_label(config: dict) -> str:
    """Short label like 'recursive_minilm_dense' or 'fixed_bm25_rerank(ce)'."""
    parts = [
        config.get("chunking_strategy", "?"),
        config.get("embedding_model") or "bm25",
        config.get("retriever_type", "?"),
    ]
    label = "_".join(parts)
    if config.get("use_reranking"):
        rt = config.get("reranker_type", "?")
        short = "ce" if rt == "cross_encoder" else rt
        label += f"_rr({short})"
    alpha = config.get("hybrid_alpha")
    if alpha is not None and alpha != 0.7 and config.get("retriever_type") == "hybrid":
        label += f"_a{alpha}"
    return label


def _results_to_dataframe(results: list[ExperimentResult | dict]) -> pd.DataFrame:
    """Flatten ExperimentResult list into a DataFrame for plotting."""
    rows = []
    for r in results:
        if isinstance(r, dict):
            config = r["config"]
            metrics = r["metrics"]
            perf = r["performance"]
            judge = r.get("judge_scores")
            exp_id = r["experiment_id"]
        else:
            config = r.config.model_dump()
            metrics = r.metrics.model_dump()
            perf = r.performance.model_dump()
            judge = r.judge_scores.model_dump() if r.judge_scores else None
            exp_id = r.experiment_id

        row = {
            "experiment_id": exp_id,
            "label": _config_label(config),
            "chunking_strategy": config.get("chunking_strategy", ""),
            "embedding_model": config.get("embedding_model") or "bm25",
            "retriever_type": config.get("retriever_type", ""),
            "hybrid_alpha": config.get("hybrid_alpha"),
            "use_reranking": config.get("use_reranking", False),
            "reranker_type": config.get("reranker_type"),
            **{m: metrics[m] for m in _METRICS},
            "avg_query_latency_ms": perf.get("avg_query_latency_ms", 0),
            "ingestion_time_seconds": perf.get("ingestion_time_seconds", 0),
            "cost_estimate_usd": perf.get("cost_estimate_usd", 0),
        }
        if judge:
            for axis in ["avg_relevance", "avg_accuracy", "avg_completeness",
                         "avg_conciseness", "avg_citation_quality", "overall_average"]:
                row[axis] = judge.get(axis)
        rows.append(row)

    return pd.DataFrame(rows)


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save figure and close it. Returns the output path."""
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------


def plot_config_metric_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 1: All configs × 4 retrieval metrics as a heatmap."""
    with plt.style.context(_STYLE):
        pivot = df.set_index("label")[list(_METRICS)].rename(columns=_METRIC_LABELS)
        # Sort by NDCG@5 descending
        pivot = pivot.sort_values("NDCG@5", ascending=True)

        height = max(8, len(pivot) * 0.38)
        fig, ax = plt.subplots(figsize=(12, height))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    vmin=0, vmax=1, linewidths=0.5, annot_kws={"fontsize": 8})
        ax.set_title("Retrieval Metrics by Configuration", fontsize=14, pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        fig.subplots_adjust(left=0.32)
        return _save_fig(fig, output_dir, "config_metric_heatmap")


def plot_chunking_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 2 (Q1): Chunking strategy comparison, controlled for embedder+retriever."""
    with plt.style.context(_STYLE):
        # Exclude reranking and alpha-sweep variants for clean comparison
        base = df[(~df["use_reranking"]) & (df["hybrid_alpha"].isna() | (df["hybrid_alpha"] == 0.7))]
        agg = base.groupby("chunking_strategy")[_METRICS].mean().reset_index()

        # Shorten strategy names for readability
        short_names = {
            "embedding_semantic": "emb_semantic",
            "heading_semantic": "head_semantic",
            "sliding_window": "slide_window",
            "recursive": "recursive",
            "fixed": "fixed",
        }
        agg["label"] = agg["chunking_strategy"].map(lambda x: short_names.get(x, x))

        fig, axes = plt.subplots(1, 4, figsize=(16, 7))
        for i, metric in enumerate(_METRICS):
            ax = axes[i]
            bars = ax.bar(agg["label"], agg[metric],
                          color=sns.color_palette("Set2", len(agg)))
            ax.set_title(_METRIC_LABELS[metric], fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(agg)))
            ax.set_xticklabels(agg["label"], ha="right", fontsize=9, rotation=45)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("Q1: Chunking Strategy Comparison (averaged across embedders & retrievers)",
                     fontsize=13)
        fig.subplots_adjust(bottom=0.22, top=0.90, wspace=0.3)
        return _save_fig(fig, output_dir, "chunking_comparison")


def plot_embedding_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 3 (Q4): Embedding model comparison with latency + cost context."""
    with plt.style.context(_STYLE):
        # Only dense/hybrid (not BM25-only) and no reranking
        base = df[(df["embedding_model"] != "bm25") & (~df["use_reranking"])]
        agg = base.groupby("embedding_model").agg({
            **{m: "mean" for m in _METRICS},
            "avg_query_latency_ms": "mean",
            "cost_estimate_usd": "mean",
        }).reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1, 1]})

        # Panel 1: retrieval metrics — no per-bar labels (too crowded with 4 bars/group)
        x = np.arange(len(agg))
        width = 0.2
        colors = sns.color_palette("Set1", 4)
        for i, metric in enumerate(_METRICS):
            axes[0].bar(x + i * width, agg[metric], width,
                        label=_METRIC_LABELS[metric], color=colors[i])
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(agg["embedding_model"])
        axes[0].set_title("Retrieval Metrics", fontsize=12)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axes[0].grid(axis="y", alpha=0.3)

        # Panel 2: latency
        lat_colors = sns.color_palette("Set2", len(agg))
        lat_bars = axes[1].bar(agg["embedding_model"], agg["avg_query_latency_ms"],
                               color=lat_colors)
        for bar in lat_bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                         f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
        axes[1].set_title("Avg Query Latency (ms)", fontsize=12)
        axes[1].set_ylabel("ms")

        # Panel 3: cost
        cost_colors = sns.color_palette("Set3", len(agg))
        cost_bars = axes[2].bar(agg["embedding_model"], agg["cost_estimate_usd"],
                                color=cost_colors)
        for bar in cost_bars:
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                         f"${bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
        axes[2].set_title("Cost Estimate (USD)", fontsize=12)
        axes[2].set_ylabel("USD")

        fig.suptitle("Q4: Embedding Model Comparison", fontsize=13)
        fig.subplots_adjust(wspace=0.35, top=0.90)
        return _save_fig(fig, output_dir, "embedding_comparison")


def plot_retriever_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 4 (Q2): Dense vs BM25 vs Hybrid comparison."""
    with plt.style.context(_STYLE):
        base = df[~df["use_reranking"]]
        agg = base.groupby("retriever_type")[_METRICS].mean().reset_index()

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        x = np.arange(len(agg))
        width = 0.18
        colors = sns.color_palette("Set1", 4)
        for i, metric in enumerate(_METRICS):
            bars = ax.bar(x + i * width, agg[metric], width,
                          label=_METRIC_LABELS[metric], color=colors[i])
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(agg["retriever_type"])
        ax.set_ylim(0, 1.15)
        ax.set_title("Q2: Retrieval Method Comparison", fontsize=14)
        ax.legend()
        fig.tight_layout()
        return _save_fig(fig, output_dir, "retriever_comparison")


def plot_hybrid_alpha_sweep(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 5: Hybrid alpha sweep — line chart of alpha vs metrics."""
    with plt.style.context(_STYLE):
        hybrid = df[(df["retriever_type"] == "hybrid") & (df["hybrid_alpha"].notna())]
        if hybrid.empty:
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.text(0.5, 0.5, "No hybrid alpha sweep data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return _save_fig(fig, output_dir, "hybrid_alpha_sweep")

        # Group by alpha, average across chunker/embedder combos
        agg = hybrid.groupby("hybrid_alpha")[_METRICS].mean().reset_index()
        agg = agg.sort_values("hybrid_alpha")

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for metric in _METRICS:
            ax.plot(agg["hybrid_alpha"], agg[metric], "o-", label=_METRIC_LABELS[metric],
                    linewidth=2, markersize=8)
        ax.set_xlabel("Hybrid Alpha (α) — Dense Weight", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Hybrid Alpha Sweep: Dense Weight vs Retrieval Metrics", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.axvline(x=0.7, color="gray", linestyle="--", alpha=0.5, label="default α=0.7")
        fig.tight_layout()
        return _save_fig(fig, output_dir, "hybrid_alpha_sweep")


def plot_reranking_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 6 (Q3): Reranking before/after — delta bars for each base config."""
    with plt.style.context(_STYLE):
        reranked = df[df["use_reranking"]]
        if reranked.empty:
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.text(0.5, 0.5, "No reranking data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return _save_fig(fig, output_dir, "reranking_comparison")

        # Match reranked configs to their base (same chunker/embedder/retriever, no reranking)
        base = df[~df["use_reranking"]]
        pairs = []
        for _, rr_row in reranked.iterrows():
            match = base[
                (base["chunking_strategy"] == rr_row["chunking_strategy"]) &
                (base["embedding_model"] == rr_row["embedding_model"]) &
                (base["retriever_type"] == rr_row["retriever_type"])
            ]
            if not match.empty:
                base_row = match.iloc[0]
                reranker = rr_row["reranker_type"]
                short = "CE" if reranker == "cross_encoder" else reranker.upper()
                # Shorten: all reranking configs use recursive, so show embedder_retriever+reranker
                pair_label = f"{rr_row['embedding_model']}_{rr_row['retriever_type']}+{short}"
                for m in _METRICS:
                    pairs.append({
                        "config": pair_label,
                        "metric": _METRIC_LABELS[m],
                        "delta": rr_row[m] - base_row[m],
                    })

        if not pairs:
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.text(0.5, 0.5, "No matching base configs for reranking comparison",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            return _save_fig(fig, output_dir, "reranking_comparison")

        pair_df = pd.DataFrame(pairs)

        # Recall@5 and Precision@5 are always 0 (reranking reorders, doesn't change the set)
        # Show only MRR and NDCG@5 which actually change, and note the zero-delta metrics
        rank_metrics = pair_df[pair_df["metric"].isin(["MRR", "NDCG@5"])]
        zero_metrics = [m for m in pair_df["metric"].unique()
                        if pair_df[pair_df["metric"] == m]["delta"].abs().sum() == 0]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=rank_metrics, x="config", y="delta", hue="metric", ax=ax,
                    palette="Set2")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title("Q3: Reranking Impact (delta vs base config)", fontsize=14)
        ax.set_ylabel("Metric Delta (reranked − base)")
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
            label.set_fontsize(9)
        ax.legend(title="Metric", fontsize=9)
        if zero_metrics:
            fig.text(
                0.5, 0.96,
                f"Note: {', '.join(zero_metrics)} delta = 0 for all configs "
                "(reranking reorders results but doesn't change the retrieved set)",
                fontsize=9, ha="center", va="bottom", style="italic", color="gray",
            )
        fig.subplots_adjust(bottom=0.20)
        return _save_fig(fig, output_dir, "reranking_comparison")


def plot_ndcg_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 7: NDCG@5 distribution per retriever type (box plot)."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        sns.boxplot(data=df, x="retriever_type", y="ndcg_at_5", ax=ax,
                    palette="Set2", hue="retriever_type", legend=False)
        sns.stripplot(data=df, x="retriever_type", y="ndcg_at_5", ax=ax,
                      color="black", alpha=0.4, size=5)
        ax.set_title("NDCG@5 Distribution by Retrieval Method", fontsize=14)
        ax.set_ylabel("NDCG@5")
        ax.set_xlabel("Retriever Type")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        return _save_fig(fig, output_dir, "ndcg_distribution")


def plot_judge_radar(results: list[ExperimentResult | dict], output_dir: Path) -> Path:
    """Chart 8: 5-axis radar for top configs with judge scores."""
    with plt.style.context(_STYLE):
        # Filter to results with judge scores
        scored = []
        for r in results:
            if isinstance(r, dict):
                if r.get("judge_scores"):
                    scored.append(r)
            else:
                if r.judge_scores:
                    scored.append(r)

        if not scored:
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.text(0.5, 0.5, "No judge scores available\n(run with --judge to generate)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title("LLM Judge 5-Axis Radar", fontsize=14)
            return _save_fig(fig, output_dir, "judge_radar")

        # Take top 5 by NDCG@5
        def _ndcg(r):
            return r["metrics"]["ndcg_at_5"] if isinstance(r, dict) else r.metrics.ndcg_at_5
        scored.sort(key=_ndcg, reverse=True)
        top = scored[:5]

        axes_names = ["Relevance", "Accuracy", "Completeness", "Conciseness", "Citation Quality"]
        judge_keys = ["avg_relevance", "avg_accuracy", "avg_completeness",
                      "avg_conciseness", "avg_citation_quality"]

        angles = np.linspace(0, 2 * np.pi, len(axes_names), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        colors = sns.color_palette("husl", len(top))

        # Shorten labels for legend readability
        short_labels = {
            "heading_semantic": "head_sem",
            "embedding_semantic": "emb_sem",
            "sliding_window": "slide_win",
        }

        for i, r in enumerate(top):
            if isinstance(r, dict):
                js = r["judge_scores"]
                label = _config_label(r["config"])
            else:
                js = r.judge_scores.model_dump()
                label = _config_label(r.config.model_dump())

            # Shorten the label
            for long, short in short_labels.items():
                label = label.replace(long, short)

            values = [js[k] for k in judge_keys]
            values += values[:1]
            ax.plot(angles, values, "o-", color=colors[i], label=label, linewidth=2,
                    markersize=6)
            ax.fill(angles, values, color=colors[i], alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_names, fontsize=11, fontweight="bold")
        ax.set_ylim(3.0, 5.2)
        ax.set_rticks([3.5, 4.0, 4.5, 5.0])
        ax.set_title("LLM Judge 5-Axis Radar (Top 5 Configs)", fontsize=14, pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9,
                  frameon=True, framealpha=0.9)
        fig.subplots_adjust(right=0.75)
        return _save_fig(fig, output_dir, "judge_radar")


def plot_latency_vs_quality(df: pd.DataFrame, output_dir: Path) -> Path:
    """Chart 9: Scatter — avg query latency vs NDCG@5."""
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        colors = {"dense": "tab:blue", "bm25": "tab:orange", "hybrid": "tab:green"}
        for rtype, group in df.groupby("retriever_type"):
            ax.scatter(group["avg_query_latency_ms"], group["ndcg_at_5"],
                       label=rtype, color=colors.get(rtype, "gray"), s=80, alpha=0.7,
                       edgecolors="black", linewidth=0.5)

        # Annotate only the single best config overall to avoid label overlap
        best_overall = df.loc[df["ndcg_at_5"].idxmax()]
        ax.annotate(
            best_overall["label"],
            (best_overall["avg_query_latency_ms"], best_overall["ndcg_at_5"]),
            fontsize=8, ha="left", va="bottom",
            xytext=(8, 8), textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
        )

        ax.set_xlabel("Avg Query Latency (ms)", fontsize=12)
        ax.set_ylabel("NDCG@5", fontsize=12)
        ax.set_title("Latency vs Quality Trade-off", fontsize=14)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return _save_fig(fig, output_dir, "latency_vs_quality")


def plot_query_difficulty(results: list[ExperimentResult | dict], output_dir: Path) -> Path:
    """Chart 10: Per-query average NDCG@5 across all configs — shows hard queries."""
    with plt.style.context(_STYLE):
        # Collect per-query NDCG across all configs
        query_scores: dict[str, list[float]] = {}
        for r in results:
            if isinstance(r, dict):
                qrs = r.get("query_results", [])
            else:
                qrs = [qr.model_dump() for qr in r.query_results]

            for qr in qrs:
                qid = qr["query_id"] if isinstance(qr, dict) else qr.query_id
                ndcg = (qr["retrieval_scores"]["ndcg_at_5"] if isinstance(qr, dict)
                        else qr.retrieval_scores.ndcg_at_5)
                query_scores.setdefault(qid, []).append(ndcg)

        if not query_scores:
            fig, ax = plt.subplots(figsize=_FIGSIZE)
            ax.text(0.5, 0.5, "No per-query data available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return _save_fig(fig, output_dir, "query_difficulty")

        qdf = pd.DataFrame([
            {"query_id": qid, "avg_ndcg": np.mean(scores), "std_ndcg": np.std(scores)}
            for qid, scores in query_scores.items()
        ]).sort_values("avg_ndcg")

        fig, ax = plt.subplots(figsize=_FIGSIZE)
        bars = ax.barh(qdf["query_id"], qdf["avg_ndcg"], xerr=qdf["std_ndcg"],
                       color=sns.color_palette("RdYlGn", len(qdf)), capsize=3)
        ax.set_xlabel("Average NDCG@5 (across all configs)", fontsize=12)
        ax.set_ylabel("Query ID")
        ax.set_title("Per-Query Difficulty (lower = harder)", fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.75, color="red", linestyle="--", alpha=0.5, label="target (0.75)")
        ax.legend()
        fig.tight_layout()
        return _save_fig(fig, output_dir, "query_difficulty")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all_charts(
    results: list[ExperimentResult | dict],
    output_dir: str = "results/charts",
) -> list[Path]:
    """Generate all 10 charts and return their file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _results_to_dataframe(results)

    paths = [
        plot_config_metric_heatmap(df, out),
        plot_chunking_comparison(df, out),
        plot_embedding_comparison(df, out),
        plot_retriever_comparison(df, out),
        plot_hybrid_alpha_sweep(df, out),
        plot_reranking_comparison(df, out),
        plot_ndcg_distribution(df, out),
        plot_judge_radar(results, out),
        plot_latency_vs_quality(df, out),
        plot_query_difficulty(results, out),
    ]

    return paths
