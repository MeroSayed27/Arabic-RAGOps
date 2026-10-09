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


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a JSON list.")

    return data


def build_retriever() -> HybridRetriever:
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

    return HybridRetriever(
        dense_index=dense_index,
        bm25_index=bm25_index,
    )


def main() -> None:
    questions = load_questions(
        Path("evaluation/questions.json")
    )

    retriever = build_retriever()

    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    mrr_scores: list[float] = []

    print("=" * 60)
    print("Arabic-RAGOps — Hybrid Retrieval Evaluation")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print()

    for item in questions:
        results = retriever.search(
            query=item["question"],
            top_k=3,
        )

        r1 = recall_at_k(
            [
                type(
                    "Result",
                    (),
                    {"chunk": result.chunk},
                )()
                for result in results
            ],
            item["expected_chunks"],
            1,
        )

        r3 = recall_at_k(
            [
                type(
                    "Result",
                    (),
                    {"chunk": result.chunk},
                )()
                for result in results
            ],
            item["expected_chunks"],
            3,
        )

        rr = reciprocal_rank(
            [
                type(
                    "Result",
                    (),
                    {"chunk": result.chunk},
                )()
                for result in results
            ],
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