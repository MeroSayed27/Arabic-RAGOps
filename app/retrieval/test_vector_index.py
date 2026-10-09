from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.vector_index import InMemoryVectorIndex


def main() -> None:
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

    query = "ما هو نظام RAG؟"

    results = index.search(
        query=query,
        top_k=3,
    )

    print("=" * 60)
    print("Arabic-RAGOps — Semantic Retrieval Test")
    print("=" * 60)
    print(f"Model: {settings.embedding_model}")
    print(f"Indexed chunks: {index.size}")
    print(f"Query: {query}")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk.chunk_id}")
        print(f"Range: {result.chunk.start}:{result.chunk.end}")
        print("Text:")
        print(result.chunk.text)
        print("-" * 60)

    print("Semantic retrieval test completed.")


if __name__ == "__main__":
    main()