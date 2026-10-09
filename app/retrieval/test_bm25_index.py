from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.bm25_index import BM25Index


def main() -> None:
    source_file = Path("data/sample/rag_basics.md")

    document = DocumentLoader().load_file(source_file)

    chunks = TextChunker(
        chunk_size=300,
        overlap=60,
    ).split(
        document.document_id,
        document.text,
    )

    index = BM25Index()
    index.add_chunks(chunks)

    queries = [
        "ما هو نظام RAG؟",
        "What is a RAG system?",
        "ما هو دور retrieval في RAG؟",
    ]

    print("=" * 60)
    print("Arabic-RAGOps — BM25 Retrieval Test")
    print("=" * 60)
    print(f"Indexed chunks: {index.size}")
    print()

    for query in queries:
        results = index.search(query, top_k=2)

        print(f"Query: {query}")

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result.chunk.chunk_id} "
                f"(score={result.score:.4f})"
            )

        print("-" * 60)


if __name__ == "__main__":
    main() 