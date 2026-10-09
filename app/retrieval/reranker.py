from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.chunking.text_chunker import TextChunk


@dataclass(frozen=True)
class RerankedResult:
    """A chunk ranked by the cross-encoder reranker."""

    chunk: TextChunk
    score: float


class Reranker:
    """Rerank retrieved chunks using a multilingual cross-encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[TextChunk],
        top_k: int = 3,
    ) -> list[RerankedResult]:
        """Score and rerank candidate chunks."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not chunks:
            return []

        pairs = [
            [query, chunk.text]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        results = [
            RerankedResult(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]