"""Iteration log — traces every single-parameter config change with before/after metrics.

Finds config pairs differing by exactly ONE parameter, records the delta.
Produces results/iteration_log.json per PRD Section 7g.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.schemas import ExperimentConfig, ExperimentResult

# Parameters to compare — these are the config dimensions that define an experiment
_COMPARABLE_PARAMS = [
    "chunking_strategy",
    "embedding_model",
    "retriever_type",
    "hybrid_alpha",
    "use_reranking",
    "reranker_type",
]

_METRIC_KEYS = ["recall_at_5", "precision_at_5", "mrr", "ndcg_at_5"]


class IterationEntry(BaseModel):
    """One config-pair comparison where exactly one parameter differs."""

    iteration_id: int = Field(..., description="Sequential iteration number")
    parameter_changed: str = Field(..., description="Which config parameter differs")
    old_value: str = Field(..., description="Value in the 'before' config")
    new_value: str = Field(..., description="Value in the 'after' config")
    reason: str = Field(..., description="Auto-generated explanation of the change impact")
    experiment_id_before: str = Field(..., description="Experiment ID of the baseline")
    experiment_id_after: str = Field(..., description="Experiment ID of the variant")
    metrics_before: dict[str, float] = Field(..., description="Retrieval metrics for baseline")
    metrics_after: dict[str, float] = Field(..., description="Retrieval metrics for variant")
    delta: dict[str, float] = Field(..., description="Metric deltas (after - before)")


def _config_key(config: ExperimentConfig | dict, exclude_param: str) -> tuple:
    """Return a hashable key for a config, excluding one parameter."""
    if isinstance(config, dict):
        d = config
    else:
        d = config.model_dump()

    parts = []
    for p in _COMPARABLE_PARAMS:
        if p == exclude_param:
            continue
        parts.append((p, str(d.get(p, ""))))
    return tuple(parts)


def _get_metrics(result: ExperimentResult | dict) -> dict[str, float]:
    """Extract metric dict from result."""
    if isinstance(result, dict):
        m = result["metrics"]
    else:
        m = result.metrics.model_dump()
    return {k: m[k] for k in _METRIC_KEYS}


def _get_config(result: ExperimentResult | dict) -> dict:
    """Extract config dict from result."""
    if isinstance(result, dict):
        return result["config"]
    return result.config.model_dump()


def _get_experiment_id(result: ExperimentResult | dict) -> str:
    if isinstance(result, dict):
        return result["experiment_id"]
    return result.experiment_id


def build_iteration_log(
    results: list[ExperimentResult | dict],
) -> list[IterationEntry]:
    """Find config pairs differing by exactly one parameter and record deltas.

    For each parameter, groups configs that are identical on all other params,
    then pairs them. The "before" config is the one with lower NDCG@5 (so the
    delta shows improvement when positive).
    """
    if not results:
        return []

    entries: list[IterationEntry] = []
    seen_pairs: set[tuple[str, str]] = set()
    iteration_id = 1

    for param in _COMPARABLE_PARAMS:
        # Group results by all params EXCEPT this one
        groups: dict[tuple, list[ExperimentResult | dict]] = {}
        for r in results:
            key = _config_key(_get_config(r), exclude_param=param)
            groups.setdefault(key, []).append(r)

        # Within each group, pair configs that differ on this param
        for _key, group in groups.items():
            if len(group) < 2:
                continue

            # Sort by the param value so pairing is deterministic
            group.sort(key=lambda r: str(_get_config(r).get(param, "")))

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    r_a, r_b = group[i], group[j]
                    id_a = _get_experiment_id(r_a)
                    id_b = _get_experiment_id(r_b)

                    # Skip duplicate pairs
                    pair_key = tuple(sorted([id_a, id_b]))
                    if pair_key in seen_pairs:
                        continue

                    val_a = str(_get_config(r_a).get(param, ""))
                    val_b = str(_get_config(r_b).get(param, ""))
                    if val_a == val_b:
                        continue

                    seen_pairs.add(pair_key)

                    m_a = _get_metrics(r_a)
                    m_b = _get_metrics(r_b)

                    # "before" = lower NDCG@5, "after" = higher
                    if m_a["ndcg_at_5"] <= m_b["ndcg_at_5"]:
                        before, after = r_a, r_b
                        old_val, new_val = val_a, val_b
                        m_before, m_after = m_a, m_b
                    else:
                        before, after = r_b, r_a
                        old_val, new_val = val_b, val_a
                        m_before, m_after = m_b, m_a

                    delta = {k: round(m_after[k] - m_before[k], 6) for k in _METRIC_KEYS}

                    # Auto-generate reason
                    ndcg_delta = delta["ndcg_at_5"]
                    direction = "improved" if ndcg_delta > 0 else "degraded"
                    reason = (
                        f"Changing {param} from {old_val} to {new_val} "
                        f"{direction} NDCG@5 by {ndcg_delta:+.4f}"
                    )

                    entries.append(IterationEntry(
                        iteration_id=iteration_id,
                        parameter_changed=param,
                        old_value=old_val,
                        new_value=new_val,
                        reason=reason,
                        experiment_id_before=_get_experiment_id(before),
                        experiment_id_after=_get_experiment_id(after),
                        metrics_before=m_before,
                        metrics_after=m_after,
                        delta=delta,
                    ))
                    iteration_id += 1

    # Sort by absolute NDCG@5 delta descending — biggest impacts first
    entries.sort(key=lambda e: abs(e.delta.get("ndcg_at_5", 0)), reverse=True)
    # Re-number after sort
    for i, entry in enumerate(entries, 1):
        entry.iteration_id = i

    return entries


def save_iteration_log(
    entries: list[IterationEntry],
    output_path: str = "results/iteration_log.json",
) -> Path:
    """Save iteration log entries to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.model_dump(mode="json") for e in entries]
    path.write_text(json.dumps(data, indent=2))
    return path
