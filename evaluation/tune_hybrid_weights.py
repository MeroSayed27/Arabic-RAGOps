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


def build_indexes() -> tuple[InMemoryVectorIndex, BM25Index]:
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

    return dense_index, bm25_index


def evaluate(
    retriever: HybridRetriever,
    questions: list[dict],
) -> tuple[float, float, float]:
    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    mrr_scores: list[float] = []

    for item in questions:
        results = retriever.search(
            item["question"],
            top_k=3,
        )

        recall_1_scores.append(
            recall_at_k(
                results,
                item["expected_chunks"],
                1,
            )
        )

        recall_3_scores.append(
            recall_at_k(
                results,
                item["expected_chunks"],
                3,
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                results,
                item["expected_chunks"],
            )
        )

    return (
        mean(recall_1_scores),
        mean(recall_3_scores),
        mean(mrr_scores),
    )


def main() -> None:
    questions = load_questions()
    dense_index, bm25_index = build_indexes()

    weight_pairs = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]

    print("=" * 60)
    print("Arabic-RAGOps — Hybrid Weight Tuning")
    print("=" * 60)

    for dense_weight, bm25_weight in weight_pairs:
        retriever = HybridRetriever(
            dense_index=dense_index,
            bm25_index=bm25_index,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        recall_1, recall_3, mrr = evaluate(
            retriever,
            questions,
        )

        print(
            f"Dense={dense_weight:.1f} "
            f"BM25={bm25_weight:.1f}"
        )
        print(f"Recall@1: {recall_1:.4f}")
        print(f"Recall@3: {recall_3:.4f}")
        print(f"MRR:       {mrr:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()