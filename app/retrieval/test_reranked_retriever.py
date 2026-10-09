from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.reranked_retriever import RerankedRetriever
from app.retrieval.vector_index import InMemoryVectorIndex


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

    embedding_service = EmbeddingService(
        settings.embedding_model
    )

    dense_index = InMemoryVectorIndex(embedding_service)
    dense_index.add_chunks(chunks)

    bm25_index = BM25Index()
    bm25_index.add_chunks(chunks)

    hybrid = HybridRetriever(
        dense_index=dense_index,
        bm25_index=bm25_index,
    )

    reranker = Reranker()

    pipeline = RerankedRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )

    query = "ما هو نظام RAG؟"

    results = pipeline.search(
        query=query,
        retrieval_k=5,
        top_k=3,
    )

    print("=" * 60)
    print("Arabic-RAGOps — Hybrid + Reranking Test")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Retrieved candidates: 5")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. "
            f"{result.chunk.chunk_id} "
            f"(reranker score={result.score:.4f})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()