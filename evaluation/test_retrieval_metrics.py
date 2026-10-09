from __future__ import annotations

import json 
from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.document_loader import DocumentLoader
from app.retrieval.vector_index import InMemoryVectorIndex

from evaluation.retrieval_metrics import (
    mean,
    recall_at_k,
    reciprocal_rank,
)


def load_questions(path: Path) -> list[dict]:
    """Load evaluation questions from JSON."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a JSON list.")

    return data


def build_index() -> InMemoryVectorIndex:
    """Build the retrieval index from the sample knowledge base."""
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


def main() -> None:
    questions_file = Path("evaluation/questions.json")
    questions = load_questions(questions_file)
    index = build_index()

    recall_1_scores: list[float] = []
    recall_3_scores: list[float] = []
    mrr_scores: list[float] = []

    print("=" * 60)
    print("Arabic-RAGOps — Retrieval Evaluation")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print(f"Indexed chunks: {index.size}")
    print()

    for item in questions:
        question_id = item["id"]
        language = item["language"]
        question = item["question"]
        expected_chunks = item["expected_chunks"]

        results = index.search(
            query=question,
            top_k=3,
        )

        r1 = recall_at_k(
            results,
            expected_chunks,
            k=1,
        )

        r3 = recall_at_k(
            results,
            expected_chunks,
            k=3,
        )

        rr = reciprocal_rank(
            results,
            expected_chunks,
        )

        recall_1_scores.append(r1)
        recall_3_scores.append(r3)
        mrr_scores.append(rr)

        top_ids = [
            result.chunk.chunk_id
            for result in results
        ]

        print(f"Question: {question_id}")
        print(f"Language: {language}")
        print(f"Top results: {top_ids}")
        print(f"Recall@1: {r1:.4f}")
        print(f"Recall@3: {r3:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print("-" * 60)

    print()
    print("Overall Metrics")
    print("-" * 60)
    print(f"Mean Recall@1: {mean(recall_1_scores):.4f}")
    print(f"Mean Recall@3: {mean(recall_3_scores):.4f}")
    print(f"MRR: {mean(mrr_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()