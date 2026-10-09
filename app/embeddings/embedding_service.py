from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Create vector embeddings for Arabic and English text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple text strings into embedding vectors."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert one user query into an embedding vector."""
        return self.embed_texts([query])[0]