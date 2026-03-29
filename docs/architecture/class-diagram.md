# Component Class Hierarchy

Every pipeline component hides behind an ABC. Callers never import concrete classes — they call a factory function with a config string and get back the right implementation. I used the same pattern in Java with static factory methods that switch on a discriminator field, except Python enforces the contract at runtime (`@abstractmethod` raises `TypeError` on instantiation) rather than at compile time.

Splitting the diagram by component group because a single 20-class Mermaid block is unreadable.

## Chunking

Five strategies, all producing `list[Chunk]` from a `Document`. The parameters that differ between them (fixed size vs window/step vs heading detection vs embedding similarity) are the whole reason the ABC exists — the caller shouldn't care how chunks are produced, only that they conform to the `Chunk` schema.

```mermaid
classDiagram
    class BaseChunker {
        <<abstract>>
        +chunk(Document) list~Chunk~
    }
    class FixedSizeChunker {
        -chunk_size: int
        -chunk_overlap: int
        +chunk(Document) list~Chunk~
    }
    class RecursiveChunker {
        -chunk_size: int
        -chunk_overlap: int
        +chunk(Document) list~Chunk~
    }
    class SlidingWindowChunker {
        -window_size: int
        -step_size: int
        +chunk(Document) list~Chunk~
    }
    class HeadingSemanticChunker {
        -min_chunk_size: int
        +chunk(Document) list~Chunk~
    }
    class EmbeddingSemanticChunker {
        -breakpoint_threshold: float
        -min_chunk_size: int
        +chunk(Document) list~Chunk~
    }

    BaseChunker <|-- FixedSizeChunker
    BaseChunker <|-- RecursiveChunker
    BaseChunker <|-- SlidingWindowChunker
    BaseChunker <|-- HeadingSemanticChunker
    BaseChunker <|-- EmbeddingSemanticChunker
```

## Embedding

Four models spanning local CPU inference (MiniLM, mpnet), local GPU via Ollama REST API (nomic), and cloud API (OpenAI). The `dimensions` property matters because FAISS index dimension is set at creation time — you can't mix 384d and 768d vectors in the same index.

```mermaid
classDiagram
    class BaseEmbedder {
        <<abstract>>
        +embed(list~str~) ndarray
        +embed_query(str) ndarray
        +dimensions int
    }
    class MiniLMEmbedder {
        +dimensions = 384
    }
    class MpnetEmbedder {
        +dimensions = 768
    }
    class OpenAIEmbedder {
        +dimensions = 1536
    }
    class OllamaEmbedder {
        +dimensions = 768
    }

    BaseEmbedder <|-- MiniLMEmbedder
    BaseEmbedder <|-- MpnetEmbedder
    BaseEmbedder <|-- OpenAIEmbedder
    BaseEmbedder <|-- OllamaEmbedder
```

## Retrieval

The interesting piece here is `HybridRetriever`. It doesn't just inherit from `BaseRetriever` — it *composes* a `DenseRetriever` and `BM25Retriever` internally, runs both, normalizes BM25's unbounded scores to [0,1] via min-max, then fuses with `α·dense + (1-α)·bm25_norm`. The caller sees a single `retrieve()` call. This is the Composite pattern — same idea as Java's `CompositeService` where the composite owns its delegates.

```mermaid
classDiagram
    class BaseRetriever {
        <<abstract>>
        +retrieve(str, int) list~RetrievalResult~
    }
    class DenseRetriever {
        -embedder: BaseEmbedder
        -vector_store: BaseVectorStore
        +retrieve(str, int) list~RetrievalResult~
    }
    class BM25Retriever {
        -chunks: list~Chunk~
        +retrieve(str, int) list~RetrievalResult~
    }
    class HybridRetriever {
        -dense: DenseRetriever
        -bm25: BM25Retriever
        -alpha: float
        +retrieve(str, int) list~RetrievalResult~
    }

    BaseRetriever <|-- DenseRetriever
    BaseRetriever <|-- BM25Retriever
    BaseRetriever <|-- HybridRetriever
    HybridRetriever *-- DenseRetriever : composes
    HybridRetriever *-- BM25Retriever : composes
```

## Reranking, Generation, and Storage

Rerankers are the most expensive component per-query — CrossEncoder scores each (query, chunk) pair independently, so it's O(k) forward passes vs a bi-encoder's single pass. That's why reranking only applies to the top-k from retrieval, never the full corpus.

`LiteLLMClient` wraps any provider (OpenAI, Anthropic, Cohere) behind one `generate()` call. The `JSONCache` avoids re-calling the API for identical prompts during experiment reruns — saved roughly $12 across the full grid.

```mermaid
classDiagram
    class BaseReranker {
        <<abstract>>
        +rerank(str, list~RetrievalResult~, int) list~RetrievalResult~
    }
    class CrossEncoderReranker {
        +rerank(str, list~RetrievalResult~, int) list~RetrievalResult~
    }
    class CohereReranker {
        +rerank(str, list~RetrievalResult~, int) list~RetrievalResult~
    }

    class BaseLLM {
        <<abstract>>
        +generate(str, str, float) str
    }
    class LiteLLMClient {
        -model: str
        -cache: JSONCache
        +generate(str, str, float) str
    }

    class BaseVectorStore {
        <<abstract>>
        +add(list~Chunk~, ndarray) None
        +search(ndarray, int) list~tuple~
        +save(str) None
        +load(str) None
    }
    class FAISSVectorStore {
        -dimension: int
        +add(list~Chunk~, ndarray) None
        +search(ndarray, int) list~tuple~
        +save(str) None
        +load(str) None
    }

    BaseReranker <|-- CrossEncoderReranker
    BaseReranker <|-- CohereReranker
    BaseLLM <|-- LiteLLMClient
    BaseVectorStore <|-- FAISSVectorStore
```

## Factory Pattern

The factory is a Python module, not a class — five standalone functions that map config strings to concrete instances. In Java terms, this is closer to a static factory method with a `switch` on the discriminator, not Spring's `@Component` autowiring. The key property: changing `embedding_model: "minilm"` to `"openai"` in a YAML file produces a completely different pipeline with zero code changes.

```mermaid
classDiagram
    class Factory {
        <<module: factories.py>>
        +create_chunker(ExperimentConfig) BaseChunker
        +create_embedder(str) BaseEmbedder
        +create_retriever(ExperimentConfig, BaseEmbedder, list, BaseVectorStore) BaseRetriever
        +create_reranker(str) BaseReranker
        +create_llm(str, JSONCache) BaseLLM
        +load_configs(str) list~ExperimentConfig~
    }

    Factory ..> BaseChunker : creates
    Factory ..> BaseEmbedder : creates
    Factory ..> BaseRetriever : creates
    Factory ..> BaseReranker : creates
    Factory ..> BaseLLM : creates
```
