# Ingestion Pipeline

PDF documents go through four stages: extract text, split into chunks, embed into vectors, and index into FAISS. Each experiment config produces its own isolated index in `data/indices/{config_stem}/` — this prevents cross-contamination between experiment runs, so config A's index never leaks into config B's retrieval results.

The vision LLM path in extraction is optional: PyMuPDF pulls raw text from every page, but some PDFs have figures or diagrams with meaningful content. When enabled, a vision model (GPT-4o) generates text descriptions of embedded images, which get appended to the page text before chunking. I kept this off for the experiment grid (academic papers are text-heavy) but it's wired up for future use with diagram-heavy documents.

```mermaid
sequenceDiagram
    participant User
    participant CLI as ingest.py
    participant Ext as Extractor
    participant Ch as Chunker
    participant Em as Embedder
    participant VS as FAISSVectorStore
    participant Disk

    User->>CLI: python scripts/ingest.py --config {yaml}
    CLI->>CLI: Load + validate YAML → ExperimentConfig

    rect rgb(240, 248, 255)
        Note over CLI,Ext: PDF Extraction (cached to data/extracted/)
        CLI->>Ext: extract_all_pdfs(pdf_dir)
        Ext->>Ext: PyMuPDF text + optional vision LLM image descriptions
        Ext-->>CLI: list[Document]
    end

    rect rgb(245, 245, 220)
        Note over CLI,Ch: Chunking (5 strategies available)
        loop For each Document
            CLI->>Ch: chunk(document)
            Ch-->>CLI: list[Chunk]
        end
        Note right of CLI: Accumulated: all_chunks
    end

    rect rgb(240, 255, 240)
        Note over CLI,Em: Embedding (4 models available)
        CLI->>Em: embed([chunk.content for chunk in all_chunks])
        Em-->>CLI: np.ndarray (N chunks x D dimensions)
    end

    rect rgb(255, 240, 245)
        Note over CLI,Disk: Indexing + Persistence
        CLI->>VS: FAISSVectorStore(dimension=D)
        CLI->>VS: add(all_chunks, embeddings)
        CLI->>VS: save(data/indices/{config}/index)
        VS->>Disk: {config}/index.faiss + index.json
    end

    CLI-->>User: Summary: chunks, dimensions, timings, file sizes
```

## Data Flow

| Stage | Input | Output | Key Type |
|-------|-------|--------|----------|
| Extract | PDF files | `list[Document]` | `Document(id, content, metadata)` |
| Chunk | `Document` | `list[Chunk]` | `Chunk(id, content, metadata)` — metadata carries `start_char`, `end_char`, `page_number` |
| Embed | `list[str]` | `np.ndarray` | Shape: (N, D) — L2-normalized so inner product = cosine similarity |
| Index | chunks + embeddings | FAISS index | `IndexFlatIP` — exact search, not approximate. Deterministic results for reproducibility. |
| Save | index + metadata | disk files | `.faiss` (raw vectors) + `.json` (chunk content + metadata for reconstruction) |
