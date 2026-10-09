from __future__ import annotations

import json
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


def build_pipeline() -> RerankedRetriever:
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

    return RerankedRetriever(
        hybrid_retriever=hybrid,
        reranker=reranker,
    )


def main() -> None:
    questions = load_questions()
    pipeline = build_pipeline()

    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    mrr_scores: list[float] = []

    print("=" * 60)
    print("Arabic-RAGOps — Reranked Retrieval Evaluation")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print()

    for item in questions:
        results = pipeline.search(
            query=item["question"],
            retrieval_k=5,
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

        recall_1_scores.append(r1)
        recall_3_scores.append(r3)
        mrr_scores.append(rr)

        print(f"Question: {item['id']}")
        print(f"Language: {item['language']}")
        print(
            "Top results:",
            [result.chunk.chunk_id for result in results],
        )
        print(f"Recall@1: {r1:.4f}")
        print(f"Recall@3: {r3:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print("-" * 60)

    print("Overall Metrics")
    print("-" * 60)
    print(f"Mean Recall@1: {mean(recall_1_scores):.4f}")
    print(f"Mean Recall@3: {mean(recall_3_scores):.4f}")
    print(f"MRR: {mean(mrr_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()