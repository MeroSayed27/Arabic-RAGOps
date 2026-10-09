from __future__ import annotations

import json
from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_index import InMemoryVectorIndex
from evaluation.retrieval_metrics import (
    mean,
    recall_at_k,
    reciprocal_rank,
)


def load_questions() -> list[dict]:
    with Path("evaluation/questions.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_indexes() -> tuple[
    InMemoryVectorIndex,
    BM25Index,
    HybridRetriever,
]:
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

    return dense_index, bm25_index, hybrid


def evaluate(
    name: str,
    retriever,
    questions: list[dict],
) -> tuple[float, float, float]:
    recall_1: list[float] = []
    recall_3: list[float] = []
    mrr: list[float] = []

    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=3,
        )

        r1 = recall_at_k(
            results,
            item["expected_chunks"],
            1,
        )

        r3 = recall_at_k(
            results,
            item["expected_chunks"],
            3,
        )

        rr = reciprocal_rank(
            results,
            item["expected_chunks"],
        )

        recall_1.append(r1)
        recall_3.append(r3)
        mrr.append(rr)

    return (
        mean(recall_1),
        mean(recall_3),
        mean(mrr),
    )


def main() -> None:
    questions = load_questions()
    dense, bm25, hybrid = build_indexes()

    results = {
        "Dense": evaluate("Dense", dense, questions),
        "BM25": evaluate("BM25", bm25, questions),
        "Hybrid": evaluate("Hybrid", hybrid, questions),
    }

    print("=" * 60)
    print("Arabic-RAGOps — Retrieval Method Comparison")
    print("=" * 60)

    for name, metrics in results.items():
        recall_1, recall_3, mrr = metrics

        print(name)
        print(f"  Recall@1: {recall_1:.4f}")
        print(f"  Recall@3: {recall_3:.4f}")
        print(f"  MRR:       {mrr:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()