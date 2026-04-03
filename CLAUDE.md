# CLAUDE.md — P5: ShopTalk Knowledge Management Agent

> **Read this file + docs/PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P5 — ShopTalk Knowledge Management Agent: Production RAG System
- **Location:** `rubsj/05-shoptalk-knowledge-agent` (standalone repo, extracted from ai-portfolio monorepo)
- **Timeline:** 7 sessions (~30h total), revised plan with learning/experiment days prioritizing depth
- **PRD:** `docs/PRD.md` — the product requirements contract (v4, hardware upgrade + expanded timeline)
- **Concepts Primer:** `docs/learning/concepts-primer.html` — deep reference for chunking, hybrid retrieval, score fusion, NDCG, cross-encoders, ABC pattern, citation extraction

---

## Model Routing Protocol (CRITICAL)

**Opus plans. Sonnet executes.**

### Opus (Planning & Architecture)
- Start of each day: read PRD tasks, create detailed implementation plan with file-by-file approach
- Design Pydantic schemas, ABC interfaces, function signatures
- Debug non-trivial issues (conceptual, not typos)
- Analyze experiment results and decide what findings matter
- Any ambiguity in the PRD

### Sonnet (Implementation)
- All code writing — implement what Opus planned
- File creation, dependency setup, test writing
- Running commands (uv sync, pytest, experiment runs)
- Routine fixes (imports, parameters, formatting)
- Chart generation, documentation

### Session Workflow
```
1. Opus: "Read CLAUDE.md and docs/PRD.md. Today is Day [N]. Plan implementation."
2. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
3. Sonnet: "Execute the plan. Start with [first file]."
4. Sonnet implements, tests, commits
5. If blocked → Opus for debugging
6. Session end → Sonnet for git commit, journal entry, CLAUDE.md update
```

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1–P4.
- **Learning Priority:** Learning depth over speed. Rabbit holes encouraged if they produce genuine insight.
- **Hardware:** Mac Pro M5 Max, 128GB unified RAM, 40 GPU cores. (Upgraded from M2 8GB — see Hardware section below.)
- **IDE:** VS Code + Claude Code extension

### Patterns Proven in P1–P4 (Reuse These)
| Pattern | Source | P5 Application |
|---------|--------|----------------|
| Pydantic models + validators | P1/P4 | Document, Chunk, RetrievalResult, ExperimentConfig |
| Instructor + auto-retry | P1/P4 | Ground truth generation, LLM-as-Judge |
| JSON file cache (MD5 key) | P1/P2/P4 | Cache LLM responses + embeddings |
| FAISS + SentenceTransformers | P2 | Vector indexing with explicit persistence |
| Model lifecycle: load → use → del → gc.collect() | P2/P3 | Still preferred for clean benchmarking (not a survival constraint) |
| Retrieval metrics (Precision, Recall, MRR) | P2 | Plus NDCG@K (new) |
| matplotlib/seaborn charts | P1–P4 | 11+ experiment comparison charts |
| Click CLI | P2 | 3 commands: ingest, serve, evaluate |
| Rich progress bars | P2 | Batch processing progress |
| ADR template | P1–P4 | 7 ADRs (distributed across Days 1-5) |

### New for P5 (Learn These)
- **PyMuPDF** (`import fitz`) — PDF text extraction + page rendering. `fitz.open()` = Java's `PDDocument.load()`. `page.get_text("dict")` returns text AND image blocks with bounding boxes. Vision LLM (GPT-4o-mini) describes figures/tables, interleaved at correct position (ADR-006).
- **rank-bm25** — `BM25Okapi(tokenized_corpus)`. Corpus = list of token lists, NOT embeddings.
- **LiteLLM** — `from litellm import completion`. Wraps OpenAI/Anthropic/Cohere behind one API. Like Java's adapter pattern.
- **NDCG** — Implement from scratch. `DCG@K = Σ(rel_i / log₂(i+1))`. `NDCG = DCG / ideal_DCG`.
- **Score fusion** — ALWAYS normalize BM25 [0,∞) to [0,1] via min-max before combining with cosine [0,1].
- **CrossEncoder** — `from sentence_transformers import CrossEncoder`. Takes (query, doc) pairs. Slower but more accurate than bi-encoder.
- **Embedding-semantic chunking** — Cosine similarity between consecutive sentence embeddings. Split where similarity drops. ALWAYS use MiniLM for boundary detection (PRD Decision 1).
- **tiktoken** — `enc = tiktoken.get_encoding("cl100k_base")` for accurate token counting. Like Java's `StringTokenizer` but for LLM tokens.
- **YAML + Pydantic** — `yaml.safe_load()` → Pydantic model. NEVER `yaml.load()` (security risk like Java's `ObjectInputStream` deserialization).
- **ABC pattern** — `from abc import ABC, abstractmethod`. Python's interface equivalent. `@abstractmethod` = Java `abstract`. Enforced at runtime, not compile time.
- **Ollama REST API** — `httpx.post("http://localhost:11434/api/embeddings", json={...})`. Local embedding model inference. Zero API cost. (v4 addition)

---

## Writing Rules

> Inherited from portfolio-wide standards. Applies to all docs — ADRs, journal entries, READMEs, code comments.

- Write as a practitioner documenting real decisions, not a consultant producing a deliverable
- First person is allowed and preferred where natural ("I picked X because", "this burned us")
- Never narrate the document's own importance — if it mattered, just state what happened
- No section whose only purpose is to make the author look good
- Analogies go inline as parentheticals — never in their own dedicated section
- Bold emotional category labels ("Easier:", "Harder:") are banned — write plain prose or plain bullets
- Numbers and benchmarks stay where they're contextually relevant — never aggregate into a "Validation" section
- Section headers are plain nouns — not action phrases, not corporate labels
- If a sentence could have been written without knowing anything specific about this project, delete it
- Code comments explain WHY, never what — if the code is readable, no comment needed
- No hedging openers in comments: ban "Note that", "This ensures", "It's worth mentioning"
- Docstrings: one sentence what + one sentence non-obvious how/why — no parameter narration
- Inline comments for short context, block comments only for genuinely non-obvious decisions
- Comment like you're explaining to a teammate at 11pm — direct, no filler

---

## Architecture Rules (FINAL — Do Not Re-Debate)

These come from PRD Sections 3 and 5. All design decisions are finalized.

1. **FAISS** — not ChromaDB. Explicit persistence (index + metadata files).
2. **No LangChain** — build from ABCs. First-principles RAG.
3. **LiteLLM** — multi-provider wrapper behind BaseLLM. Default to OpenAI models.
4. **Instructor** — structured output for ground truth + judge. `max_retries=3`, `mode=instructor.Mode.JSON`.
5. **5 chunking strategies** — Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic.
6. **4 embedding models** — MiniLM (384d), mpnet (768d), OpenAI text-embedding-3-small (1536d), nomic-embed-text via Ollama (768d). *(v4: added Ollama)*
7. **3 retrieval methods** — Dense, BM25, Hybrid (α-weighted, min-max normalized).
8. **2 rerankers** — Cohere API + local CrossEncoder. Both behind BaseReranker.
9. **5-axis LLM-as-Judge** — Relevance, Accuracy, Completeness, Conciseness, Citation Quality. NOT RAGAS.
10. **NDCG from scratch** — understand the math. 4-level grading: 3=gold, 2=same section, 1=same doc, 0=irrelevant.
11. **YAML + Pydantic** for experiment configs.
12. **Click CLI** (ingest, serve, evaluate) + **Streamlit** web UI. NO FastAPI.
13. **Hybrid ground truth** — LLM generates 30, developer curates 15. 1-3 gold chunks per query.
14. **35 experiment configs** in core grid (5 chunkers × 3 embedders × 2 retrieval methods + 5 BM25 baselines).
15. **Index-reference citations** — [N] markers parsed to chunk objects. Parse-only validation for grid; judge handles semantic validation.
16. **System performance metrics** — track ingestion time, query latency, index size, peak memory.
17. **Braintrust** — optional stretch. JSON baseline first.
18. **Embedding-semantic chunker** — ALWAYS uses MiniLM for boundary detection, regardless of indexing embedder.
19. **HybridRetriever** — internal composition. Owns both DenseRetriever and BM25Retriever.
20. **Experiment runner** — pools experiments by embedding model. MiniLM first, mpnet second, BM25, Ollama nomic, then OpenAI.
21. **Test coverage** — ≥95% on all core modules. pytest --cov enforced.
22. **(v4) OllamaEmbedder** — implements BaseEmbedder via Ollama REST API. Health check on init. Graceful skip if Ollama not running.

---

## Hardware Context (v4 — M5 Max 128GB)

**Upgraded from:** MacBook Air M2, 8GB RAM
**Current machine:** Mac Pro M5 Max, 128GB unified memory, 40 GPU cores, ~800 GB/s memory bandwidth

### What This Changes for P5

| Dimension | Old (M2 8GB) | New (M5 Max 128GB) | Impact |
|-----------|-------------|---------------------|--------|
| Concurrent models | ONE at a time or OOM | Multiple models simultaneously | Pooling is for clean benchmarking, not survival |
| CrossEncoder loading | Load/unload per reranking pass | Keep loaded alongside everything else | Simpler reranking pipeline |
| FAISS index size | Constrained to <10K vectors | Millions of vectors fit in RAM | No index size limitations |
| Local LLM inference | Impossible | Ollama with 7B-70B models | Enables local embedding experiments |
| Batch embedding | Reduced batches (50) to survive | Full batches (500+) | Faster ingestion |
| psutil monitoring | Critical survival mechanism | Informational only | Remove OOM panic paths |

### What This Does NOT Change
- **Pooling by embedder:** Still the right approach for clean, comparable benchmarks. Not a constraint anymore, but a discipline.
- **del + gc.collect():** Still good practice between experiment groups. Not critical, but keeps baseline memory clean.
- **Architecture decisions:** FAISS, no LangChain, ABC pattern — all unchanged. These were design choices, not hardware workarounds.
- **Cache strategy:** Still cache all LLM and embedding calls to avoid regeneration cost.

### Ollama Setup (Day 0 prerequisite)
```bash
# Install Ollama (if not already installed)
brew install ollama

# Start Ollama server (runs in background)
ollama serve &

# Pull embedding model for P5
ollama pull nomic-embed-text

# Verify it's working
curl http://localhost:11434/api/embeddings -d '{"model": "nomic-embed-text", "prompt": "test"}'
# Should return JSON with "embedding": [0.123, ...]
```

---

## Memory Management Protocol (128GB M5 Max)

> **v4: Rewritten from survival protocol to benchmarking discipline.**
> With 128GB, RAM pressure is no longer a constraint. The protocol below exists for clean benchmarking and good engineering practice — not OOM prevention.

```
GUIDELINE 1: Pool experiments by embedder for clean benchmarks.
  Load MiniLM → run all MiniLM configs → unload → load mpnet → repeat.
  WHY: Ensures all configs for one model run under identical conditions.
  This is about benchmarking integrity, not memory survival.

GUIDELINE 2: Clean up between experiment groups.
  del model + gc.collect() between embedder groups.
  WHY: Prevents stale references from affecting measurements.

GUIDELINE 3: Ollama runs out-of-process.
  OllamaEmbedder calls localhost:11434 — Ollama manages its own memory.
  No Python-side model lifecycle management needed.

GUIDELINE 4: FAISS indices can stay loaded.
  128GB means all indices (even 10K+ vectors at 1536d) fit comfortably.
  No need to load/unload indices between experiments.

GUIDELINE 5: OpenAI embeddings = API calls = zero local RAM for that model.
  Batch up to 500 texts per call. Respect rate limits.

OPTIONAL MONITORING: import psutil; psutil.virtual_memory().percent
  No longer critical, but useful for documenting performance metrics in experiment results.
```

---

## Session Plan Mapping (v4 → Original PRD)

> The revised plan adds 2 days to the original 5-day PRD scope. This table maps each revised day to the PRD task scope to prevent confusion during Claude Code sessions.

| Revised Day | Focus | Original PRD Day | What Changed |
|-------------|-------|-------------------|-------------|
| Day 1 | Foundation: Schemas, ABCs, extraction, chunking | PRD Day 1 | **Already complete.** Verify and update CLAUDE.md state. |
| Day 2 | Retrieval Pipeline: Embedders, FAISS, retrievers, rerankers, generator | PRD Day 2 | Resume here (Mar 19). Same scope as original Day 2. |
| Day 3 | Evaluation Framework: Metrics, ground truth, judge, configs, runner | PRD Day 3 | Same scope. Add Ollama config YAMLs to experiment grid. |
| Day 4 | Experiment Execution: Full grid, charts, comparison report (Q1-Q4) | PRD Day 4 | Expanded time (Sunday deep work). More configs with Ollama added. |
| Day 5 | **NEW:** Local Model Experiments + Concept Deep-Dive | — | OllamaEmbedder implementation, local vs API comparison (Q5), ADR-007. |
| Day 6 | Streamlit UI + CLI Polish | PRD Day 5 (split) | UI and CLI only. Documentation moved to Day 7. |
| Day 7 | Documentation Sprint: README, Loom, Concept Library, Journal | PRD Day 5 (split) | Dedicated documentation day for gold-standard depth. |

---

## Notion Integration

| Resource | ID / URL |
|----------|----------|
| Command Center | `https://www.notion.so/2ffdb630640a81f58df5f5802aa51550` |
| Requirements Doc | `https://www.notion.so/Mini_Project_5_requirements-2ffdb630640a81668269f59ead053417` |
| Requirements Traceability | `https://www.notion.so/Requirements-Traceability-312db630640a816fa2e0cee9dc7ba829` |
| Project Tracker (data source) | `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` |
| P5 Tracker Card | *(create on Day 1 — update this field)* |
| Learning Journal (data source) | `collection://c707fafc-4c0e-4746-a3bc-6fc4cd962ce5` |
| ADR Log (data source) | `collection://629d4644-ca7a-494f-af7c-d17386e1189b` |
| Chat Index | `303db630640a81ccb026f767597b023f` |

### Journal Entry Template

**Notion properties to set:**
- `Entry`: `P5 Day [N] — [Short descriptive title]`
- `Project`: `P5: ShopTalk Knowledge Agent`
- `Phase`: one of `Foundation | Build | Evaluate | Experiment | Document | Polish`
- `Session Type`: one of `Weeknight | Sunday Deep Work | Saturday`
- `Hours`: numeric
- `Date`: YYYY-MM-DD
- `Python Pattern Learned`: 2–5 bullet summary (inline text field)
- `Blocked By`: brief description, or blank

**Page body template:**

```markdown
## What I shipped
For each component built this session: what it does, what file it lives in, and why I designed
it the way I did. Focus on the non-obvious calls — anyone can read the code for the obvious parts.
Include specific implementation details that would trip me up if I came back cold in two weeks.

## Numbers
Whatever I actually measured: pass rates, error counts, latency, API cost, test coverage.
No invented precision — if I didn't measure it, I don't include it.
Before/after if something changed. One line per metric is fine.

## What I actually learned
### [Name the concept]
Write this as if explaining to past-me from two months ago. Full paragraphs, not bullets.
Cover: what it is, why it matters beyond this specific project, and where it maps to
something I already know from Java/TS. One subsection per real insight — not one per topic touched.

## What blocked me
### [Name the blocker] — RESOLVED / ONGOING
- What broke and how I found it
- The actual root cause (not just the symptom)
- What let it get this far without being caught
- What I did to fix it
- The principle I'm taking forward

## Python pattern of the day
### [Name it]
The one Python or library pattern worth a deep dive. Show the real code with WHY comments.
Explain what's happening under the hood, why this over the obvious alternative, and the
Java/TS equivalent side-by-side. Cover the parameters that actually matter and why I set them
the way I did.

## Next session
Concrete tasks, not intentions. Reference PRD task IDs where applicable.
```

### ADR Template

**Notion properties to set:**
- `Decision`: `ADR-NNN: [Title]`
- `Project`: `P5: ShopTalk Knowledge Agent`
- `Category`: one of `Data Model | Architecture | Tool Choice | Algorithm | Deployment | Evaluation`
- `Status`: `Accepted | Superseded | Proposed`
- `Date`: YYYY-MM-DD

**Page body template:**

```markdown
## Context
What situation forced this decision? Be specific about the constraints — scale, failure modes,
prior project lessons. Reference real incidents (e.g., "this burned us in P4 because...").
Write in first person where it's natural. Skip background the team already knows.

## Decision
One sentence stating what was chosen. Then explain how it actually works — the mechanism,
not the marketing. Include the exact code/config used, because that's what future-me needs.

## Alternatives Considered

| Option | Trade-off | Why rejected |
|--------|-----------|--------------|
| **Chosen** ✅ | what you give up | — selected |
| Alternative A | what it offers vs. costs | specific reason, ideally from experience |
| Alternative B | what it offers vs. costs | specific reason, ideally from experience |

## What the numbers said
Any benchmarks, error rates, latency measurements, or cost figures that informed the call.
State them inline as facts, not as a proof section. If you didn't measure it, don't invent precision.

## What this changes
Plain bullets on what gets easier, what gets harder, and which future projects (P6–P9) can reuse this.
No bold category labels — just write it out.

## Cross-References
- Links to other ADRs this depends on or affects
```

---

## Code Conventions

### Python Style
- **Type hints on ALL function signatures and return types**
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (Python 3.12 native generics)
- Use `str | None` over `Optional[str]`
- Use `@classmethod` with `@field_validator` (Pydantic v2 pattern)
- Use `model_validate_json()` not `parse_raw()` (deprecated in v2)
- Use `model_json_schema()` to generate schemas for LLM prompts
- Prefer dataclasses for simple data containers, Pydantic when validation needed
- Use `pathlib.Path` over `os.path`
- Use f-strings for prompts, logs, paths
- **Pydantic for all data** — no raw dicts crossing module boundaries
- **yaml.safe_load()** — NEVER `yaml.load()`
- **Pool experiments by embedder** — for benchmarking integrity
- **≥95% test coverage** — pytest --cov enforced on every module

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Comments
- **WHY not what** — `# WHY: min-max normalize because BM25 ranges [0,∞) vs cosine [0,1]`
- No hedging openers: ban "Note that", "This ensures", "It's worth mentioning"
- Docstrings: one sentence what + one sentence non-obvious how/why — no parameter narration
- Inline comments for short context, block comments only for genuinely non-obvious decisions

### Error Handling
- Use specific exception types, never bare `except:`
- Log errors with context (what operation, what input caused it)
- For LLM calls: always wrap in try/except, implement retry with backoff

### Testing
- pytest, not unittest
- Test file mirrors source: `src/schemas.py` → `tests/test_schemas.py`
- Test names: `test_<what>_<when>_<expected>` (e.g., `test_chunk_when_empty_doc_raises_validation_error`)
- Include both happy path and failure cases
- Use parametrize for multiple inputs

---

## Git Conventions

- **NEVER commit directly to `main`**. Always work on a feature branch and merge via PR.
- Branch naming: `type/scope-short-description` (e.g., `feat/p5-hybrid-retriever`, `fix/p5-bm25-normalization`)
- Commit messages: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
  - Scope: `p5`
  - Example: `feat(p5): add hybrid retriever with min-max score fusion`
- Commit after each logical unit of work (not at end of night)
- Workflow: create branch → commit → push → create PR → merge to main

---

## LLM Cost Management

- Cache ALL LLM responses during development
- Cache key: `hashlib.md5(f"{model}\n{system_prompt}\n---\n{user_prompt}".encode()).hexdigest()`
- Cache location: `data/cache/` as JSON files
- Before any LLM call, check cache first
- Log estimated cost per call to console
- GPT-4o-mini: ~$0.15/1M input, ~$0.60/1M output
- GPT-4o: ~$2.50/1M input, ~$10.00/1M output

---

## Session End Protocol

At the end of every session, Sonnet must:

1. **Git commit and push** all work
2. **Update Current State** below (check off completed tasks)
3. **Write journal entry** to Notion Learning Journal
4. **Produce handoff summary:**

```
## P5 Handoff — Session End [Date]

### Branch / Commit
### What's Done (PRD task references)
### Key Files Created/Modified
### Key Metrics (if measured)
### What's Next
### Key Learnings (the interview story)
### Blockers / Open Questions
```

---

## Current State

> **UPDATE at the end of every session.**

### Day 0 (Pre-start)
- [x] Project directory created
- [x] pyproject.toml with all dependencies
- [x] `uv sync` passes
- [x] .env configured (OPENAI_API_KEY, COHERE_API_KEY)
- [x] P5 card created in Notion Project Tracker
- [x] Test PDFs downloaded to `data/pdfs/`
- [x] Ollama installed and nomic-embed-text pulled (v4 — verified 2026-03-21, 768d confirmed)

### Day 1 — Foundation: Extraction + Chunking + Schemas ✅ COMPLETE
- [x] All Pydantic schemas (matching requirements data models + PRD additions)
- [x] All 6 ABCs (BaseChunker, BaseEmbedder, BaseVectorStore, BaseRetriever, BaseReranker, BaseLLM)
- [x] PDF extraction with PyMuPDF + text cleaning + vision LLM image descriptions (ADR-006)
- [x] 5 chunking strategies (Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic)
- [x] Tests for schemas + all chunkers (≥95% coverage)
- [x] **ADR-001: FAISS over ChromaDB** written and committed
- [x] **ADR-002: No LangChain — first-principles RAG** written and committed
- [x] **Checkpoint:** All chunkers produce valid Chunks from test PDFs.

> **Day 1 Note (v4):** Day 1 was completed prior to the March pause. Code exists and passes tests. Resume from Day 2 on March 19. Before starting Day 2, verify: `uv sync` still passes, tests still green, schemas match PRD v4 (should be compatible — v4 adds OllamaEmbedder but doesn't change existing schemas).

> **Day 0 re-verified 2026-03-21.** All dependencies synced on M5 Max 128GB. 162/162 tests green. Overall coverage 97% (all core modules ≥95%). Ollama running with nomic-embed-text (768d). BaseEmbedder ABC abstract methods: `embed`, `embed_query`, `dimensions`. Ready for Day 2.

### Day 2 — Retrieval Pipeline ← RESUME HERE (Mar 19)
- [ ] 2 local embedders (MiniLM, mpnet)
- [ ] FAISSVectorStore with save/load persistence
- [ ] Dense retriever
- [ ] BM25 retriever
- [ ] Hybrid retriever with min-max normalization + α weighting
- [ ] Cohere reranker + CrossEncoder reranker
- [ ] LiteLLM generator with citation extraction
- [ ] Factory pattern: config string → class
- [ ] End-to-end smoke test: PDF → chunk → embed → index → retrieve → generate → cited answer
- [ ] Tests for all new components (≥95% coverage)
- [ ] **ADR-003: Hybrid retrieval with min-max score fusion** written and committed
- [ ] **ADR-004: LiteLLM over raw OpenAI SDK** written and committed
- [ ] **Checkpoint:** Full query pipeline works end-to-end.

### Day 3 — Evaluation + Experiment Setup
- [ ] Retrieval metrics from scratch (Precision@K, Recall@K, MRR, NDCG@K)
- [ ] Ground truth: LLM generates 30, developer curates 15
- [ ] 5-axis LLM-as-Judge via Instructor
- [ ] OpenAI embedder (3rd model)
- [ ] 35+ YAML experiment configs (including Ollama nomic placeholders for Day 5)
- [ ] Experiment runner (grid orchestrator pooled by embedder)
- [ ] System performance metric collection (ingestion time, latency, index size, memory)
- [ ] Tests for metrics + judge (≥95% coverage)
- [ ] **ADR-005: YAML + Pydantic for experiment configs** written and committed
- [ ] **Checkpoint:** Evaluation framework complete. Ready for big run.

### Day 4 — Run Experiments + Analysis (Sunday Deep Work) ✅ COMPLETE
- [x] Full experiment grid run (46 configs with judge, excluding Ollama)
- [x] 10 visualization charts (Q1-Q4, alpha sweep, judge radar, latency scatter, query difficulty)
- [x] Comparison report answering 4 required questions (Q1-Q4) — 11 sections, 317 lines
- [x] Extended experiments (α sweep 0.3/0.5/0.7/0.9, 8 reranking configs: cross_encoder + cohere)
- [x] Judge calibration (5 pairs scored by human and LLM, saved to results/judge_calibration_input.json)
- [x] Pipeline orchestrator (scripts/evaluate.py with --no-judge, --reproducibility-check flags)
- [x] **(v5)** Iteration log: 114 entries with before/after metrics + delta → `results/iteration_log.json`
- [x] **(v5)** Final config traceability table in comparison report → `results/comparison_report.md`
- [x] **(v5)** Reproducibility verification: 0% variance on all 4 metrics (deterministic pipeline)
- [x] **Checkpoint:** 46 configs complete. Best config (heading_semantic_openai_dense): NDCG@5=0.896, Recall@5=1.000, MRR=0.907, Judge=4.77/5.0. 3/4 retrieval targets met (Precision@5 missed — ceiling from GT density). Iteration log + reproducibility verified.

> **Day 4 Key Results:**
> - Best config: `heading_semantic_openai_dense` (experiment ID: `470e2e37`)
> - PRD 2a: Recall@5=1.000 (PASS), MRR=0.907 (PASS), NDCG@5=0.896 (PASS), Precision@5=0.300 (FAIL — GT density ceiling)
> - PRD 2b: Judge overall=4.77 (PASS, target >4.0)
> - 562 tests, 94% coverage
> - Branch: `feat/p5-day4-experiments`, PR #8

### Day 5 — Local Model Experiments + Concept Deep-Dive (NEW in v4) ✅ COMPLETE
- [x] OllamaEmbedder implementation (implements BaseEmbedder, calls localhost:11434)
- [x] Verify Ollama + nomic-embed-text running locally
- [x] Run Ollama nomic configs (best 3 chunkers × 2 retrieval methods = 6 configs)
- [x] Local vs API embedding comparison chart (Q5) — Chart 11
- [x] Latency benchmarking: local embedding time vs OpenAI API time
- [x] **ADR-007: Local vs API embeddings** written and committed
- [x] Concept Library entries: "Cross-encoder Reranking", "Hybrid Search" (Notion only)
- [x] Learning Journal: deep-dive reflection on experimentation insights (Notion)
- [x] **Checkpoint:** Q5 answered. Best Ollama config: sliding_window+hybrid NDCG@5=0.757 (-0.139 vs OpenAI best). 576 tests, ≥94% coverage. PR #10 merged.

> **Day 5 Key Results:**
> - Best Ollama config: `sliding_window_ollama_nomic_hybrid` (NDCG@5=0.757, Recall@5=0.889, MRR=0.722, $0 cost)
> - 3 of 6 Ollama configs match or exceed equivalent mpnet. Widest gap: −0.065 vs mpnet sliding_window+hybrid.
> - httpx added as dependency. OllamaUnavailableError + graceful skip pattern implemented.
> - 576 tests, coverage ≥94%. Branch: `feat/p5-day5-ollama`, PR #10.

### Day 6 — Streamlit UI + CLI Polish ✅ COMPLETE
- [x] Click CLI (ingest, serve, evaluate) — all converted/implemented with Click decorators
- [x] `src/vector_store.py` — added `chunks` read-only property for BM25/hybrid retriever
- [x] `scripts/ingest.py` — PDF → chunk → embed → FAISS index, Rich progress, Ollama guard
- [x] `scripts/serve.py` — interactive QA REPL, cited answers, Ollama guard
- [x] `scripts/evaluate.py` — argparse → Click conversion, all flags preserved
- [x] `src/streamlit_app.py` — 5-panel app: upload, config, process, Q&A, source viewer
- [x] `tests/test_cli.py` — CliRunner tests for all 3 scripts (15 tests, all mocked)
- [x] `tests/test_streamlit_app.py` — helper function tests (15 tests, all mocked)
- [x] **Checkpoint:** 606 tests, 94% coverage (src/). Branch: `feat/p5-day6-ui`.

### Day 7 — Documentation Sprint ✅ COMPLETE
- [x] README.md (gold standard: results, architecture Mermaid diagrams, findings, ADR table, tech stack, quick start, known gaps)
- [x] Architecture diagrams: 4 Mermaid files in docs/architecture/ (class, ingestion, query, system context)
- [x] Self-evaluation questions answered (6 from PRD v4) in docs/self-evaluation.md
- [x] Final ≥95% coverage verified: 627 tests, 97% coverage (up from 94%)
- [x] PRD.md updated: D11 description + directory structure for docs/architecture/
- [ ] Loom recording (2-min walkthrough) — batched separately
- [x] Final git push + PR
- [x] **P5 COMPLETE**

---

## Troubleshooting Guide

### "BM25 scores dominate hybrid retrieval"
BM25 scores [0,∞) vs cosine [0,1]. Apply min-max normalization to BM25 BEFORE combining with α. Verify: both arrays in [0,1] after normalization. Handle edge case: all BM25 scores identical → division by zero → set all to 0.0.

### "FAISS IndexFlatIP returns low similarity scores"
IndexFlatIP = inner product. Equals cosine similarity ONLY if vectors are L2-normalized. Call `faiss.normalize_L2(embeddings)` before add() AND before search(). SentenceTransformers normalizes by default — verify with `np.linalg.norm(emb)` ≈ 1.0.

### "Embedding-semantic chunker is slow"
It embeds every sentence during chunking. Uses MiniLM (smallest, per PRD Decision 1). If still too slow, fall back to heading-semantic for that run.

### "CrossEncoder too slow for full corpus"
CrossEncoder is O(n) per candidate. ONLY use for reranking top-20, NOT full retrieval. Pipeline: bi-encoder → top-20 → cross-encoder → top-5.

### "LLM judge gives all 5s"
Leniency bias. Manual calibration: score 5 answers yourself, compare. Add negative examples to judge prompt.

### "YAML config validation error at runtime"
This is why Pydantic validates at load time. Common: missing hybrid_alpha when method=hybrid, missing reranker when use_reranking=True.

### "No config meets Recall@5 > 0.80 target"
Iterate: increase chunk overlap, try reranking, try larger embedder. The debugging story is valuable in interviews regardless. Document what you tried.

### "PyMuPDF two-column PDF text interleaved"
Use `page.get_text("dict")` for coordinate-based extraction with bounding boxes, sort by column position. NOTE: `get_text("blocks")` silently omits image blocks — always use `get_text("dict")` when you need image positions. Or: choose single-column technical docs.

### "Figures/tables missing from extracted content"
PyMuPDF's `get_text()` only extracts text. Use `describe_images=True` in `extract_pdf()` to render pages as PNG and send to GPT-4o-mini vision for figure descriptions. Results are cached to `data/extracted/{stem}.json` — subsequent calls are free. See ADR-006.

### "LiteLLM import error or routing issue"
LiteLLM model strings: `"gpt-4o-mini"` (not `"openai/gpt-4o-mini"` unless explicitly routing). Check `litellm.model_list` for supported models. Fallback: raw OpenAI SDK behind BaseLLM if LiteLLM is problematic.

### "Ollama connection refused" (v4)
Ollama server must be running: `ollama serve &`. Verify: `curl http://localhost:11434/api/tags`. If model not pulled: `ollama pull nomic-embed-text`. OllamaEmbedder has health check in `__init__` — will raise clear error if Ollama is unreachable.

### "Ollama embedding dimensions don't match FAISS index" (v4)
nomic-embed-text produces 768d vectors — same as mpnet. FAISS index dimension is set at creation time. If reusing an existing index built with MiniLM (384d), you MUST create a new index for Ollama/mpnet configs. The experiment runner handles this — each (chunker, embedder) pair gets its own index.
