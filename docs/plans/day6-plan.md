# Day 6 Plan — Streamlit UI + CLI Polish

## Context
Day 5 (PR #11) merged to main with 576 tests, ≥94% coverage. All pipeline components (extraction, chunkers, embedders, retrievers, rerankers, generator, factories, experiment_runner) are complete. Day 6 builds the user-facing CLI and Streamlit interfaces that compose these existing components. No new pipeline logic — pure UI/CLI shell around proven code.

**Branch:** `feat/p5-day6-ui`

---

## Deliverables

### 1. `scripts/evaluate.py` — argparse → Click conversion
- Convert existing 158-line argparse script to Click decorators
- Preserve all functionality: `--configs`, `--ground-truth`, `--output`, `--pdfs`, `--no-judge`, `--reproducibility-check`
- Keep all output formatting and summary table logic identical
- Click option mapping: `click.Path(exists=True, path_type=Path)` for directories, `is_flag=True` for booleans

### 2. `scripts/ingest.py` — Click CLI (rewrite from stub)
- `--config` (required, YAML path) + `--pdf-dir` (default `data/pdfs/`)
- Flow: `extract_all_pdfs()` → `create_chunker(config).chunk()` → `create_embedder().embed()` → `FAISSVectorStore.add()` → `.save()`
- Index saved to `data/indices/{config.stem}/index` → produces `index.faiss` + `index.json`
- Rich progress bars for extract/chunk/embed/index stages
- BM25-only configs (no `embedding_model`) → clear error message, not a crash
- Reuses: `src.extraction.extract_all_pdfs`, `src.factories.create_chunker/create_embedder`, `src.vector_store.FAISSVectorStore`, `src.schemas.ExperimentConfig`

### 3. `scripts/serve.py` — Click CLI REPL (rewrite from stub)
- `--config` (required, YAML path) + `--model` (default `gpt-4o-mini`)
- Startup: load config → create embedder → load FAISS index from `data/indices/{config.stem}/index` → create retriever → create reranker (if configured) → create LLM
- REPL loop: question → `retriever.retrieve()` → optional `reranker.rerank()` → `build_qa_prompt()` → `llm.generate()` → `extract_citations()` → display answer + sources
- `exit`/`quit`/`q` to stop
- Reuses: `src.factories.*`, `src.generator.build_qa_prompt/extract_citations`, `src.vector_store.FAISSVectorStore`

### 4. `src/vector_store.py` — Add `chunks` property
- Add `@property chunks -> list[Chunk]` to `FAISSVectorStore` to expose loaded chunks for `create_retriever()` (which needs chunks for BM25/hybrid)
- Currently only accessible via `_chunks` (private)

### 5. `src/streamlit_app.py` — 5-panel Streamlit app (rewrite from stub)

**Sidebar (Panel 1+2):**
- `st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)`
- Chunking strategy selectbox + conditional params (chunk_size/overlap for fixed/recursive, window/step for sliding_window, etc.)
- Embedding model selectbox: minilm, mpnet, ollama_nomic, openai
- Retriever selectbox: dense, bm25, hybrid (+ alpha slider if hybrid)
- Top K slider (1–20, default 5)
- Reranking checkbox + reranker selectbox
- "Process Documents" button → triggers ingestion pipeline

**Processing (on button click):**
- Build `ExperimentConfig` from sidebar selections
- Save uploaded PDFs to tempdir → `extract_pdf()` each
- Chunk → embed → build FAISS index
- Store everything in `st.session_state` (documents, chunks, embedder, vector_store, config, index_ready flag)
- Ollama error handling: catch connection errors, show `st.error()` with instructions

**Main area (Panel 3+4+5):**
- `st.text_input` for question + Submit button
- Answer display with `st.markdown(answer)` + latency caption
- Source viewer: `st.expander` per citation showing chunk content + page number

**Extracted helper functions (for testability):**
- `build_config_from_ui(**kwargs) → ExperimentConfig` — maps UI selections to validated config
- `run_query(query, config, embedder, chunks, vector_store) → dict` — executes retrieve → rerank → generate → cite pipeline, returns {answer, citations, latency_ms, chunks_used}

### 6. Tests

**`tests/test_cli.py`** — Click CliRunner tests:
- `TestIngestCli`: creates index files (mocked), fails on missing config, fails on BM25-only config
- `TestServeCli`: fails on missing index, loads and exits on "quit" input (mocked)
- `TestEvaluateCli`: help shows options, fails on missing configs dir

**`tests/test_streamlit_app.py`** — Helper function tests:
- `TestBuildConfigFromUi`: valid dense config, BM25 sets embedding=None, hybrid requires alpha, reranking requires type
- `TestRunQuery`: returns answer and citations (all pipeline components mocked)
- `TestStreamlitAppImport`: module imports without error

---

## Implementation Order
1. `scripts/evaluate.py` (Click conversion) — smallest change, establishes Click pattern
2. `src/vector_store.py` (add `chunks` property) — prerequisite for serve.py and streamlit
3. `scripts/ingest.py` (full implementation)
4. `scripts/serve.py` (full implementation)
5. `src/streamlit_app.py` (full implementation with extracted helpers)
6. `tests/test_cli.py`
7. `tests/test_streamlit_app.py`

## Critical Files
- `scripts/evaluate.py` — rewrite (argparse → Click)
- `scripts/ingest.py` — rewrite from stub
- `scripts/serve.py` — rewrite from stub
- `src/streamlit_app.py` — rewrite from stub
- `src/vector_store.py` — add `chunks` property
- `tests/test_cli.py` — new file
- `tests/test_streamlit_app.py` — new file

## Verification
1. `python scripts/ingest.py --config experiments/configs/01_fixed_minilm_dense.yaml` → index files appear in `data/indices/01_fixed_minilm_dense/`
2. `python scripts/serve.py --config experiments/configs/01_fixed_minilm_dense.yaml` → REPL loads, answers a question, exits on "quit"
3. `python scripts/evaluate.py --help` → shows all Click options
4. `streamlit run src/streamlit_app.py` → app loads, upload PDF, configure, ask question, get cited answer
5. `pytest tests/test_cli.py tests/test_streamlit_app.py -v` → all pass
6. `pytest --cov=src --cov=scripts --cov-report=term-missing` → ≥94% coverage maintained
