from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_index import InMemoryVectorIndex


def build_indexes() -> tuple[InMemoryVectorIndex, BM25Index]:
    source_file = Path("data/sample/rag_basics.md")

    document = DocumentLoader().load_file(source_file)

    chunks = TextChunker(
        chunk_size=300,
        overlap=60,
    ).split(
        document.document_id,
        document.text,
    )

    embedding_service = EmbeddingService(
        settings.embedding_model
    )

    dense_index = InMemoryVectorIndex(embedding_service)
    dense_index.add_chunks(chunks)

    bm25_index = BM25Index()
    bm25_index.add_chunks(chunks)

    return dense_index, bm25_index


def main() -> None:
    dense_index, bm25_index = build_indexes()

    hybrid = HybridRetriever(
        dense_index=dense_index,
        bm25_index=bm25_index,
    )

    queries = [
        "ما هو نظام RAG؟",
        "What is a RAG system?",
        "ما هو دور retrieval في RAG؟",
    ]

    print("=" * 60)
    print("Arabic-RAGOps — Hybrid Retrieval Test")
    print("=" * 60)

    for query in queries:
        results = hybrid.search(
            query=query,
            top_k=3,
        )

        print(f"Query: {query}")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result.chunk.chunk_id} "
                f"(RRF score={result.score:.6f})"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()