"""Streamlit web UI for interactive RAG question-answering.

5 panels:
  1. PDF upload — drag-and-drop, triggers ingestion pipeline
  2. Config sidebar — chunker, embedder, retriever, reranker selection
  3. Question input — free-text query box
  4. Answer + citations — generated answer with [N] citations expanded
  5. Source viewer — retrieved chunks with relevance scores and page numbers

Why Streamlit over FastAPI+React: spec-required. P4 proved FastAPI skills.
P5's value is the RAG pipeline, not another REST wrapper.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from src.embedders.ollama_embedder import OllamaUnavailableError
from src.extraction import extract_pdf
from src.factories import create_chunker, create_embedder, create_llm, create_reranker, create_retriever
from src.generator import build_qa_prompt, extract_citations
from src.schemas import ExperimentConfig
from src.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Extracted helpers (pure functions — testable without Streamlit)
# ---------------------------------------------------------------------------


def build_config_from_ui(
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str | None,
    retriever_type: str,
    hybrid_alpha: float | None,
    top_k: int,
    use_reranking: bool,
    reranker_type: str | None,
    window_size_tokens: int | None = None,
    step_size_tokens: int | None = None,
) -> ExperimentConfig:
    """Map UI widget selections to a validated ExperimentConfig.

    Raises pydantic.ValidationError if the combination is invalid (e.g.,
    hybrid without alpha, reranking without reranker_type).
    """
    return ExperimentConfig.model_validate(
        {
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "retriever_type": retriever_type,
            "hybrid_alpha": hybrid_alpha,
            "top_k": top_k,
            "use_reranking": use_reranking,
            "reranker_type": reranker_type,
            "window_size_tokens": window_size_tokens,
            "step_size_tokens": step_size_tokens,
        }
    )


def run_query(
    query: str,
    config: ExperimentConfig,
    embedder,
    chunks,
    vector_store: FAISSVectorStore,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """Execute the full RAG pipeline for a single query.

    Returns dict with keys: answer, citations, latency_ms, chunks_used.
    All components are passed in so this function is fully testable via mocks.
    """
    retriever = create_retriever(config, embedder, chunks, vector_store)
    reranker = create_reranker(config.reranker_type) if config.use_reranking else None
    llm = create_llm(model=llm_model)

    start = time.monotonic()

    results = retriever.retrieve(query, top_k=config.top_k)
    if reranker is not None:
        results = reranker.rerank(query, results, top_k=config.top_k)

    context_chunks = [r.chunk for r in results]
    prompt = build_qa_prompt(query, context_chunks)
    answer = llm.generate(prompt)
    citations = extract_citations(answer, context_chunks)

    latency_ms = (time.monotonic() - start) * 1000

    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": latency_ms,
        "chunks_used": context_chunks,
    }


# ---------------------------------------------------------------------------
# Streamlit page layout
# ---------------------------------------------------------------------------


def _render_sidebar() -> dict:  # pragma: no cover
    """Render sidebar config panels and return a dict of widget values."""
    st.sidebar.header("Configuration")

    # --- Panel 1: PDF Upload ---
    st.sidebar.subheader("1. Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # --- Panel 2: Pipeline Config ---
    st.sidebar.subheader("2. Pipeline Settings")

    chunking_strategy = st.sidebar.selectbox(
        "Chunking strategy",
        options=["fixed", "recursive", "sliding_window", "heading_semantic", "embedding_semantic"],
        index=1,
    )

    chunk_size = 512
    chunk_overlap = 50
    window_size_tokens = None
    step_size_tokens = None

    if chunking_strategy in ("fixed", "recursive"):
        chunk_size = st.sidebar.slider("Chunk size (chars)", min_value=100, max_value=2000, value=512, step=50)
        chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", min_value=0, max_value=200, value=50, step=10)
    elif chunking_strategy == "sliding_window":
        window_size_tokens = st.sidebar.slider("Window size (tokens)", min_value=50, max_value=1000, value=200, step=50)
        step_size_tokens = st.sidebar.slider("Step size (tokens)", min_value=10, max_value=500, value=100, step=10)

    retriever_type = st.sidebar.selectbox(
        "Retriever",
        options=["dense", "bm25", "hybrid"],
        index=0,
    )

    embedding_model: str | None = None
    hybrid_alpha: float | None = None

    if retriever_type in ("dense", "hybrid"):
        embedding_model = st.sidebar.selectbox(
            "Embedding model",
            options=["minilm", "mpnet", "openai", "ollama_nomic"],
            index=0,
        )

    if retriever_type == "hybrid":
        hybrid_alpha = st.sidebar.slider(
            "Hybrid alpha (dense weight)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        )

    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=20, value=5)

    use_reranking = st.sidebar.checkbox("Enable reranking")
    reranker_type: str | None = None
    if use_reranking:
        reranker_type = st.sidebar.selectbox(
            "Reranker",
            options=["cross_encoder", "cohere"],
        )

    llm_model = st.sidebar.selectbox(
        "LLM model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    process_clicked = st.sidebar.button("Process Documents", type="primary", disabled=not uploaded_files)

    return {
        "uploaded_files": uploaded_files,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "retriever_type": retriever_type,
        "hybrid_alpha": hybrid_alpha,
        "top_k": top_k,
        "use_reranking": use_reranking,
        "reranker_type": reranker_type,
        "window_size_tokens": window_size_tokens,
        "step_size_tokens": step_size_tokens,
        "llm_model": llm_model,
        "process_clicked": process_clicked,
    }


def _process_documents(ui: dict) -> None:  # pragma: no cover
    """Ingest uploaded PDFs and store pipeline state in session_state."""
    with st.spinner("Processing documents..."):
        try:
            cfg = build_config_from_ui(
                chunking_strategy=ui["chunking_strategy"],
                chunk_size=ui["chunk_size"],
                chunk_overlap=ui["chunk_overlap"],
                embedding_model=ui["embedding_model"],
                retriever_type=ui["retriever_type"],
                hybrid_alpha=ui["hybrid_alpha"],
                top_k=ui["top_k"],
                use_reranking=ui["use_reranking"],
                reranker_type=ui["reranker_type"],
                window_size_tokens=ui["window_size_tokens"],
                step_size_tokens=ui["step_size_tokens"],
            )
        except Exception as e:
            st.error(f"Invalid configuration: {e}")
            return

        # Save uploaded PDFs to a temp dir and extract
        documents = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for uploaded_file in ui["uploaded_files"]:
                pdf_path = Path(tmpdir) / uploaded_file.name
                pdf_path.write_bytes(uploaded_file.read())
                try:
                    doc = extract_pdf(str(pdf_path))
                    documents.append(doc)
                except Exception as e:
                    st.error(f"Failed to extract {uploaded_file.name}: {e}")
                    return

        if not documents:
            st.error("No documents extracted.")
            return

        # Chunk
        chunker = create_chunker(cfg)
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunker.chunk(doc))

        # Embed
        try:
            embedder = create_embedder(cfg.embedding_model)
        except OllamaUnavailableError:
            st.error(
                "Ollama is not running. Start it with:\n\n"
                "```\nollama serve\nollama pull nomic-embed-text\n```"
            )
            return

        embeddings = embedder.embed([c.content for c in all_chunks])

        # Index
        store = FAISSVectorStore(dimension=embedder.dimensions)
        store.add(all_chunks, embeddings)

        # Persist to session_state
        st.session_state["config"] = cfg
        st.session_state["embedder"] = embedder
        st.session_state["chunks"] = all_chunks
        st.session_state["vector_store"] = store
        st.session_state["index_ready"] = True
        st.session_state["doc_count"] = len(documents)
        st.session_state["chunk_count"] = len(all_chunks)

    st.success(
        f"Processed {st.session_state['doc_count']} document(s) → "
        f"{st.session_state['chunk_count']} chunks. Ready to answer questions."
    )


def main() -> None:  # pragma: no cover
    """Entry point — renders the full 5-panel Streamlit app."""
    st.set_page_config(page_title="ShopTalk Knowledge Agent", layout="wide")
    st.title("ShopTalk Knowledge Agent")
    st.caption("RAG-powered Q&A over your PDF documents")

    ui = _render_sidebar()

    # Trigger ingestion when button is clicked
    if ui["process_clicked"]:
        _process_documents(ui)

    # Show index status
    if st.session_state.get("index_ready"):
        st.info(
            f"Index ready: {st.session_state['chunk_count']} chunks "
            f"from {st.session_state['doc_count']} document(s)"
        )
    else:
        st.warning("Upload PDFs and click **Process Documents** to get started.")
        return

    # --- Panel 3: Question Input ---
    st.subheader("Ask a Question")
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("Your question", placeholder="What are the return policy requirements?", label_visibility="collapsed")
    with col2:
        submit = st.button("Submit", type="primary")

    if not submit or not question.strip():
        return

    # --- Panels 4+5: Answer + Source Viewer ---
    with st.spinner("Thinking..."):
        try:
            result = run_query(
                query=question.strip(),
                config=st.session_state["config"],
                embedder=st.session_state["embedder"],
                chunks=st.session_state["chunks"],
                vector_store=st.session_state["vector_store"],
                llm_model=ui["llm_model"],
            )
        except Exception as e:
            st.error(f"Query failed: {e}")
            return

    # Panel 4: Answer
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.caption(f"Latency: {result['latency_ms']:.0f} ms")

    # Panel 5: Source Viewer
    if result["citations"]:
        st.subheader("Sources")
        for i, cit in enumerate(result["citations"], 1):
            source_name = Path(cit.source).name
            page = cit.page_number + 1  # 0-indexed internally
            with st.expander(f"[{i}] {source_name} — page {page}"):
                # Find the full chunk text for this citation
                chunk_text = next(
                    (c.content for c in result["chunks_used"] if c.id == cit.chunk_id),
                    cit.text_snippet,
                )
                st.markdown(f"**Source:** `{source_name}`")
                st.markdown(f"**Page:** {page}")
                st.text(chunk_text)


if __name__ == "__main__":
    main()
