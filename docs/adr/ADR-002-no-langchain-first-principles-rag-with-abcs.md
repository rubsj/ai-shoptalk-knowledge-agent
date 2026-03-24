# ADR-002: No LangChain: First-Principles RAG with ABCs

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Architecture
**Status:** Accepted
**Date:** 2026-03-03

---

## Context

LangChain provides `RecursiveCharacterTextSplitter`, `RetrievalQA`, `HuggingFaceEmbeddings`, and dozens of pre-built chains for RAG pipelines.

I didn't use it. P5 is an experiment platform that compares chunking strategies and retriever types. The experiment variables are the implementation details LangChain hides. I need to control chunk boundaries, measure where splits happen, and swap components without fighting a framework.

---

## Decision

Build all RAG components from scratch behind Abstract Base Classes. No LangChain dependency.

Six ABCs in `src/interfaces.py` define the contract: `BaseChunker`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`, `BaseReranker`, and `BaseGenerator`. Each one isolates a single experiment variable. Five chunking strategies extend `BaseChunker`, ranging from a fixed-size character window baseline to an `EmbeddingSemanticChunker` that detects boundaries via cosine similarity breakpoints.

The key design constraint: every implementation must be independently testable without loading real models or real PDFs. `EmbeddingSemanticChunker` is tested with a mocked `SentenceTransformer` that crafts orthogonal embeddings to place boundaries exactly where the test expects them. `RecursiveChunker._split_text()` takes any text + separator list and returns segments directly, no splitter object needed. LangChain's equivalent is a class method coupled to the full splitter.

---

## Alternatives Considered

**LangChain** - Pre-built components, fast scaffolding. Hides internals so you can't inspect why a boundary was placed. Version churn. Hard to unit-test without mocking LangChain's own guts. It abstracts away the exact things I need to measure.

**Haystack** - RAG-focused pipeline abstractions. Same problem: abstracts the experiment variables.

**LlamaIndex** - Good document indexing. Wraps chunking and retrieval into node/pipeline abstractions. Same problem.

---

## Consequences

Every component is independently benchmarkable. Swapping a chunker in an experiment is one line: the `ExperimentConfig.chunking_strategy` field. 53 unit tests cover all five chunkers, and none load real PDFs or real models, because there are no third-party abstractions to stub out.

More boilerplate per component, and I have to handle retry and error logic myself since there's no `Runnable` infrastructure.

The `BaseChunker` / `BaseEmbedder` / `BaseRetriever` pattern is reusable across P6 through P9. If I ever migrate to LangChain for a production project, it's an adapter implementation, not a rewrite. The ABC pattern maps directly to Java interfaces (defining your own `Splitter` interface instead of depending on Spring's `TextSplitter` bean).
