from __future__ import annotations

from app.config.settings import settings
from app.embeddings.embedding_service import EmbeddingService


def main() -> None:
    service = EmbeddingService(settings.embedding_model)

    examples = [
        ("Arabic", "ما هو نظام RAG؟"),
        ("English", "What is a RAG system?"),
        ("Mixed", "ما هو RAG system؟"),
    ]

    print("=" * 60)
    print("Arabic-RAGOps — Embedding Test")
    print("=" * 60)
    print(f"Model: {settings.embedding_model}")
    print()

    for language, text in examples:
        vector = service.embed_query(text)

        print(f"Language: {language}")
        print(f"Text: {text}")
        print(f"Dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        print("-" * 60)


if __name__ == "__main__":
    main()