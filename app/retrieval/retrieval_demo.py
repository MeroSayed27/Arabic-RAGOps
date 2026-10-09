from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.vector_index import InMemoryVectorIndex


def build_index() -> InMemoryVectorIndex:
    """Build a semantic index from the sample knowledge base."""
    source_file = Path("data/sample/rag_basics.md")

    loader = DocumentLoader()
    document = loader.load_file(source_file)

    chunker = TextChunker(
        chunk_size=300,
        overlap=60,
    )

    chunks = chunker.split(
        document_id=document.document_id,
        text=document.text,
    )

    embedding_service = EmbeddingService(
        settings.embedding_model
    )

    index = InMemoryVectorIndex(embedding_service)
    index.add_chunks(chunks)

    return index


def run_query(
    index: InMemoryVectorIndex,
    query: str,
    top_k: int = 3,
) -> None:
    """Run one semantic retrieval query."""
    results = index.search(
        query=query,
        top_k=top_k,
    )

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result.chunk.chunk_id} "
            f"(score={result.score:.4f})"
        )
        print(result.chunk.text)
        print("-" * 60)


def main() -> None:
    index = build_index()

    queries = [
        "ما هو نظام RAG؟",
        "What is a RAG system?",
        "ما هو دور retrieval في RAG؟",
    ]

    for query in queries:
        run_query(
            index=index,
            query=query,
            top_k=2,
        )


if __name__ == "__main__":
    main()