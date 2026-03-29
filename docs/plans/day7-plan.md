# P5 Day 7 — Documentation Sprint Plan

## Context

Day 6 is complete (PR #12 merged, commit `cf9b015`, 606 tests, 94% coverage). Day 7 is the final session: create architecture diagrams, write a gold-standard README, answer 6 self-evaluation questions with real experiment data, push coverage to >=95%, and mark P5 complete. Branch `feat/p5-day7-docs` already created.

**Phase 1 findings (already gathered):**
- Best config: `heading_semantic_openai_dense` — NDCG@5=0.896, Recall@5=1.0, MRR=0.907, Judge=4.77
- Reproducibility: 0% variance on all 4 metrics
- `results/summary.json` does not exist — all data from `results/comparison_report.md`
- Coverage gaps: `extraction.py` (64%), `cohere_reranker.py` (78%), `experiment_runner.py` (93%), `iteration_log.py` (94%), `reporting.py` (94%)
- Need ~26 more lines covered to hit 95% overall (2199 stmts, 136 miss → ~110)

---

## Phase 2A: PRD Update + Architecture Diagrams (STOP after)

### PRD.md Updates

The PRD currently says D11 is "architecture (Mermaid diagram)" — singular. Update to reflect the 4 architecture diagrams:

1. **Section 7f, D11 row** (`PRD.md:482`): Change content from `Problem, architecture (Mermaid diagram), results, experiment findings, demo link, setup` to `Problem, architecture (4 Mermaid diagrams: class hierarchy, ingestion sequence, query sequence, system context), results, experiment findings, setup`

2. **Section 9 directory structure** (`PRD.md:609`): Add `docs/architecture/` directory under `docs/`:
   ```
   └── docs/
       ├── adr/                          ← Architecture Decision Records
       └── architecture/                 ← Mermaid architecture diagrams
   ```

Create `docs/architecture/` with 4 Mermaid markdown files. These must be done **before** the README so the README can reference/embed them.

### 1. `docs/architecture/class-diagram.md` — ABC hierarchy + factory

Mermaid `classDiagram` showing:
- **BaseChunker** (abstract `chunk(Document): list[Chunk]`)
  - FixedSizeChunker, RecursiveChunker, SlidingWindowChunker, HeadingSemanticChunker, EmbeddingSemanticChunker
- **BaseEmbedder** (abstract `embed(list[str]): ndarray`, `embed_query(str): ndarray`, property `dimensions: int`)
  - MiniLMEmbedder (384d), MpnetEmbedder (768d), OpenAIEmbedder (1536d), OllamaEmbedder (768d)
- **BaseRetriever** (abstract `retrieve(str, int): list[RetrievalResult]`)
  - DenseRetriever, BM25Retriever, HybridRetriever (composes Dense + BM25)
- **BaseReranker** (abstract `rerank(str, list[RetrievalResult], int): list[RetrievalResult]`)
  - CrossEncoderReranker, CohereReranker
- **BaseLLM** (abstract `generate(str, str, float): str`)
  - LiteLLMClient
- **BaseVectorStore** (abstract `add()`, `search()`, `save()`, `load()`)
  - FAISSVectorStore
- **Factory functions:** `create_chunker()`, `create_embedder()`, `create_retriever()`, `create_reranker()`, `create_llm()` — arrows from factory to ABCs

Key methods only, no parameters — keep scannable. Group by component type.

Source files:
- `src/interfaces.py` — all 6 ABCs
- `src/factories.py` — factory functions
- `src/chunkers/` — 5 implementations
- `src/embedders/` — 4 implementations
- `src/retrievers/` — 3 implementations (hybrid.py composes Dense+BM25)
- `src/rerankers/` — 2 implementations
- `src/generator.py` — LiteLLMClient
- `src/vector_store.py` — FAISSVectorStore

### 2. `docs/architecture/ingestion-sequence.md` — PDF-to-index flow

Mermaid `sequenceDiagram` showing:
```
CLI (ingest.py) → Extractor: extract_all_pdfs(pdf_dir)
Extractor → CLI: list[Document]
CLI → Chunker: chunk(document)
Chunker → CLI: list[Chunk]
CLI → Embedder: embed(texts)
Embedder → CLI: np.ndarray (N x D)
CLI → FAISSVectorStore: add(chunks, embeddings)
CLI → FAISSVectorStore: save(path)
FAISSVectorStore → Disk: .faiss + .json files
```
Show data types at each step.

Source: `scripts/ingest.py` lines 49-140

### 3. `docs/architecture/query-sequence.md` — Question-to-answer flow

Mermaid `sequenceDiagram` showing:
```
User → CLI (serve.py): question
CLI → Embedder: embed_query(question)
Embedder → CLI: query_vector
CLI → Retriever: retrieve(query, top_k)
Retriever → CLI: list[RetrievalResult]
opt Reranking enabled
  CLI → Reranker: rerank(query, results, top_k)
  Reranker → CLI: reranked list[RetrievalResult]
end
CLI → Generator: build_qa_prompt() + generate()
Generator → CLI: answer text
CLI → CitationExtractor: extract_citations(answer, chunks)
CitationExtractor → CLI: list[Citation]
CLI → User: answer + cited sources
```

Source: `scripts/serve.py` lines 107-151

### 4. `docs/architecture/system-context.md` — Entry points diagram

Mermaid `graph TD` showing:
- 3 CLI scripts: `ingest.py`, `serve.py`, `evaluate.py`
- 1 Streamlit app: `streamlit_app.py`
- Each maps to pipeline components they use
- External deps: OpenAI API, Cohere API, Ollama (localhost:11434)
- Data stores: `data/pdfs/`, `data/indices/`, `data/cache/`, `results/`

Source: `scripts/ingest.py`, `scripts/serve.py`, `scripts/evaluate.py`, `src/streamlit_app.py`

---

## Phase 2B: README.md + Self-Evaluation (STOP after)

### README.md — Full Rewrite
**File:** `README.md`

Inverted pyramid structure (no emoji in headers, no TOC, no placeholder badges):

1. **Title + one-line description**
   - `# ShopTalk Knowledge Management Agent`
   - Production RAG system for academic paper Q&A with 46-configuration experiment grid

2. **Key Results table**
   | Metric | Value | Target |
   | NDCG@5 | 0.896 | >0.75 |
   | Recall@5 | 1.000 | >0.80 |
   | MRR | 0.907 | >0.70 |
   | LLM Judge | 4.77/5.0 | >4.0 |
   Best config: `heading_semantic_openai_dense`. Reproducibility: 0% variance.

3. **Architecture section**
   - Embed ingestion + query sequence diagrams inline (Mermaid renders on GitHub)
   - Link to class diagram and system context files (too large for inline)

4. **Experiment Findings** (Q1-Q5 as prose with numbers):
   - Q1: heading_semantic best chunker (NDCG@5=0.7752 avg)
   - Q2: hybrid beats dense (+0.034 NDCG@5 avg)
   - Q3: reranking +0.1124 avg NDCG@5, all 8 configs improved
   - Q4: OpenAI best embedder (NDCG@5=0.8288), MiniLM fastest (8ms)
   - Q5: Ollama competitive with mpnet at $0, -0.14 gap vs OpenAI

5. **ADR table** — 7 rows linking to `docs/adr/`:
   | # | Title | File |
   | ADR-001 | FAISS over ChromaDB | docs/adr/ADR-001-faiss-over-chromadb-for-experiment-vector-index.md |
   | ADR-002 | No LangChain — First-Principles RAG with ABCs | ... |
   | ADR-003 | Hybrid Retrieval with Min-Max Score Fusion | ... |
   | ADR-004 | LiteLLM over Raw OpenAI SDK | ... |
   | ADR-005 | YAML + Pydantic Experiment Configs | ... |
   | ADR-006 | PDF Extraction with Vision LLM Image Descriptions | ... |
   | ADR-007 | Local vs API Embeddings (Ollama nomic-embed-text) | ... |

6. **Tech stack table** — PyMuPDF, FAISS, SentenceTransformers, LiteLLM, Instructor, rank-bm25, Cohere, Ollama, Streamlit, Click, Pydantic, Matplotlib/Seaborn

7. **Quick start**
   ```
   git clone <repo> && cd ai-shoptalk-knowledge-agent
   uv sync
   cp .env.example .env  # Add OPENAI_API_KEY, COHERE_API_KEY
   python scripts/ingest.py --config experiments/configs/01_fixed_minilm_dense.yaml
   python scripts/serve.py --config experiments/configs/01_fixed_minilm_dense.yaml
   streamlit run src/streamlit_app.py
   ```

### Self-Evaluation — `docs/self-evaluation.md` (new file)

All 6 PRD 8c questions answered with specific config IDs, metric values, and deltas from `results/comparison_report.md`:

- **Q1 (why X > Y):** `heading_semantic_openai_dense` (0.896) vs `embedding_semantic_mpnet_hybrid` (0.607). Three factors: embedder dominates (+0.24 NDCG from openai upgrade), chunker matters (+0.16 from heading_semantic), retriever type secondary.
- **Q2 (metrics vs qualitative):** Judge 4.77 aligns with top NDCG. Citation Quality (4.72) correlates with Recall@5 (1.0). Lower-NDCG configs produce answers the judge rates lower on Accuracy and Completeness.
- **Q3 (reranking top-3 vs top-5):** Only @5 metrics measured. MRR improvement (+0.1473 avg) shows first-relevant-result moves up in rank — strongest proxy for top-3 benefit. Honest note: dedicated @3 metrics would require re-running with top_k=3.
- **Q4 (citations):** Index-reference [N] format parsed by `extract_citations()` in `src/generator.py:84`. Judge Citation Quality axis = 4.72/5.0. Parse-only validation in grid; semantic validation by judge.
- **Q5 (edge cases):** 4 academic papers (up to 39K chars), 18 curated queries, BM25 empty-result handling, heading-semantic short-doc behavior. Integration tests verify end-to-end.
- **Q6 (local vs API):** Ollama best 0.757 vs OpenAI 0.896 (-0.139 gap). 3/6 Ollama configs match or exceed mpnet (same 768d). Hybrid boosts Ollama +0.09 to +0.12 NDCG@5. Recommendation: Ollama for cost-sensitive/privacy workloads; OpenAI for max quality.

---

## Phase 3: Coverage + Git + CLAUDE.md (STOP after)

### Coverage Push to >=95%

Priority targets (biggest gap first):
1. **`src/extraction.py`** (64%, 58 lines) — vision LLM paths, image description code paths
2. **`src/rerankers/cohere_reranker.py`** (78%, 8 lines) — API error handling branches
3. **`src/experiment_runner.py`** (93%, 13 lines) — edge case branches
4. **`src/iteration_log.py`** (94%, 6 lines)
5. **`src/reporting.py`** (94%, 17 lines)

Read each file's missing lines, add targeted mock-based tests. Goal: cover ~26+ additional lines to cross 95%.

### Git Workflow

Per memory rules: feature branch (already created), no direct main commits, no Co-Authored-By.

```bash
git add PRD.md docs/architecture/ README.md docs/self-evaluation.md tests/... CLAUDE.md
git commit -m "feat(p5): day 7 — documentation sprint + coverage push"
git push -u origin feat/p5-day7-docs
gh pr create --title "feat(p5): Day 7 — documentation sprint" --body "..."
```

### CLAUDE.md Update

Mark Day 7 items complete:
- [x] README.md (gold standard)
- [x] Architecture diagrams (4 Mermaid files)
- [x] Self-evaluation questions (6 answers)
- [x] Final >=95% coverage
- [x] Final git push + PR
- [x] **P5 COMPLETE**

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `PRD.md` | Update D11 description + directory structure |
| `docs/architecture/class-diagram.md` | Create new |
| `docs/architecture/ingestion-sequence.md` | Create new |
| `docs/architecture/query-sequence.md` | Create new |
| `docs/architecture/system-context.md` | Create new |
| `README.md` | Full rewrite |
| `docs/self-evaluation.md` | Create new |
| `tests/test_*.py` (TBD per coverage gaps) | Add tests |
| `CLAUDE.md` | Mark Day 7 + P5 complete |

## Verification

1. `uv run pytest --cov=src --cov-report=term-missing` — all pass, >=95%
2. Mermaid diagrams render correctly (valid syntax)
3. README embeds sequence diagrams inline, links class/system diagrams
4. Self-eval references real config IDs and metrics from comparison_report.md
5. `git diff` confirms no placeholder content
6. PR created on `feat/p5-day7-docs`
