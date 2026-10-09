from __future__ import annotations

from dataclasses import dataclass

from app.chunking.text_chunker import TextChunk
from app.embeddings.embedding_service import EmbeddingService


@dataclass(frozen=True)
class SearchResult:
    """A semantic search result for one chunk."""

    chunk: TextChunk
    score: float


class InMemoryVectorIndex:
    """Simple in-memory semantic vector index using cosine similarity."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._chunks: list[TextChunk] = []
        self._vectors: list[list[float]] = []

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed and store chunks in the index."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_service.embed_texts(texts)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Number of vectors must match number of chunks."
            )

        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the most semantically similar chunks."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not self._chunks:
            return []

        query_vector = self.embedding_service.embed_query(query)

        results: list[SearchResult] = []

        for chunk, vector in zip(self._chunks, self._vectors):
            score = self._dot_product(query_vector, vector)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    @staticmethod
    def _dot_product(
        first: list[float],
        second: list[float],
    ) -> float:
        """Calculate the dot product of two normalized vectors."""
        if len(first) != len(second):
            raise ValueError(
                "Vectors must have the same dimensions."
            )

        return sum(
            first_value * second_value
            for first_value, second_value in zip(first, second)
        )

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)