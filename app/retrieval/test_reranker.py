from __future__ import annotations

from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.reranker import Reranker


def main() -> None:
    document = DocumentLoader().load_file(
        Path("data/sample/rag_basics.md")
    )

    chunks = TextChunker(
        chunk_size=300,
        overlap=60,
    ).split(
        document.document_id,
        document.text,
    )

    query = "ما هو نظام RAG؟"

    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        chunks=chunks,
        top_k=3,
    )

    print("=" * 60)
    print("Arabic-RAGOps — Reranker Test")
    print("=" * 60)
    print(f"Model: {reranker.model_name}")
    print(f"Query: {query}")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result.chunk.chunk_id} "
            f"(score={result.score:.4f})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()