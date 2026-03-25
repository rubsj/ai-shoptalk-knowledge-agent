# ADR-005: YAML + Pydantic for Experiment Configs

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Configuration Management
**Status:** Accepted
**Date:** 2026-03-25

---

## Context

The experiment grid has 35 configurations spanning 5 chunking strategies, 3 embedding models, and 3 retriever types. Each config is a structured record with cross-field constraints: `hybrid` retriever requires `hybrid_alpha`, `dense`/`hybrid` require an `embedding_model`, `bm25` must have `embedding_model=null`, `sliding_window` requires `window_size_tokens` and `step_size_tokens`.

We need a format that:
1. Is human-editable (adding/removing configs without code changes)
2. Validates constraints at load time — not silently at runtime
3. Controls execution order (BM25 last, openai last — to batch API calls)
4. Maps string values to concrete class instances without the caller importing them

Hardcoding 35 `ExperimentConfig(...)` calls in Python would work, but swapping strategies or adding a new embedder means editing source code. Environment-variable-based config doesn't support structured grids. A database is too heavy for what is essentially a static parameter sweep.

---

## Decision

**YAML files + Pydantic validation + factory functions.**

Each config lives in `src/configs/{NN}_{chunking_strategy}_{embedder}_{retriever}.yaml`. The numeric prefix controls alphabetical (and therefore execution) order without any additional ordering logic.

```yaml
# src/configs/11_fixed_minilm_hybrid.yaml
chunking_strategy: fixed
chunk_size: 512
chunk_overlap: 50
top_k: 5
use_reranking: false
embedding_model: minilm
retriever_type: hybrid
hybrid_alpha: 0.7
```

`load_configs(config_dir)` uses `yaml.safe_load()` exclusively — no `yaml.load()` with a full `Loader`, no `yaml.unsafe_load()`. `safe_load` only handles standard YAML scalars, sequences, and mappings; it cannot execute arbitrary Python. `ExperimentConfig.model_validate()` runs Pydantic cross-field validators immediately after, so invalid combos (hybrid without alpha, bm25 with embedding_model) raise at load time before any API credits are spent.

The factory layer (`create_chunker`, `create_embedder`, `create_retriever`) maps the validated string values to concrete instances. Callers never import `RecursiveChunker` directly — they call `create_chunker(config)`. Adding a new chunking strategy means: add a YAML value to the `ChunkingStrategy` Literal in `schemas.py`, add a branch in `create_chunker`, add a new config file. No other files change.

---

## Alternatives Considered

**Python dataclasses / raw dicts** — No validation. Cross-field constraints become runtime assertions scattered through the pipeline. A bad combo (hybrid, no alpha) only fails when `HybridRetriever` tries to use the alpha value, not at load time.

**JSON** — Equally valid for machine reads, but no comments, no multi-line values, and harder to edit by hand. Maintaining 35 JSON files without comments for the `why` behind each config is painful.

**Dynaconf / Hydra** — Both support structured configs and validation. Hydra adds `@hydra.main` decorator coupling and its own config composition system. Dynaconf is production-oriented with secrets management. Both are more infrastructure than a 35-file parameter sweep needs. Pydantic + plain YAML achieves the same validation with zero framework lock-in.

**Single YAML file with a list** — Simpler file count, but harder to add/remove individual configs, harder to see at a glance which config is which, and alphabetical ordering by filename is lost.

---

## Quantified Validation

- 35 configs load and validate in under 10ms (`load_configs('src/configs')`).
- `yaml.safe_load()` prevents arbitrary code execution — important since configs may eventually be user-supplied or CI-generated.
- Cross-field validation catches 5 classes of invalid combos at load time:
  - `hybrid` without `hybrid_alpha`
  - `hybrid_alpha` set on non-hybrid retriever
  - `dense`/`hybrid` without `embedding_model`
  - `bm25` with non-null `embedding_model`
  - `sliding_window` without `window_size_tokens`/`step_size_tokens`

---

## Consequences

Adding a new experiment dimension (e.g., a new reranker) requires: a new YAML Literal value, a new factory branch, and new config files. The factory is the only place that imports concrete classes; all other modules stay interface-clean.

The `{NN}_` prefix is a convention, not enforced by code. A misnamed file still loads; it just sorts differently. This is an acceptable trade-off — the naming convention is self-documenting and the test suite asserts count and structural invariants, catching accidental duplicates or missing files.

The placeholder `data/ground_truth/ground_truth.json` contains `{"queries": []}`, which intentionally fails `GroundTruthSet` validation (Pydantic `min_length=1`). This forces human curation of ground truth before experiments can run, preventing accidental execution against an empty dataset.
