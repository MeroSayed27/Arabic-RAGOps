from __future__ import annotations

from dataclasses import dataclass

from app.chunking.text_chunker import TextChunk
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vector_index import InMemoryVectorIndex


@dataclass(frozen=True)
class HybridSearchResult:
    """A combined dense + BM25 retrieval result."""

    chunk: TextChunk
    score: float


class HybridRetriever:
    """Combine dense and BM25 retrieval using normalized scores."""

    def __init__(
        self,
        dense_index: InMemoryVectorIndex,
        bm25_index: BM25Index,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> None:
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if dense_weight + bm25_weight <= 0:
            raise ValueError(
                "At least one weight must be greater than 0."
            )

        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        """Normalize scores to the 0-1 range."""
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0 for _ in scores]

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[HybridSearchResult]:
        """Return results using weighted normalized-score fusion."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        dense_results = self.dense_index.search(
            query=query,
            top_k=top_k,
        )

        bm25_results = self.bm25_index.search(
            query=query,
            top_k=top_k,
        )

        dense_scores = self._normalize_scores(
            [result.score for result in dense_results]
        )

        bm25_scores = self._normalize_scores(
            [result.score for result in bm25_results]
        )

        scores: dict[str, float] = {}
        chunks: dict[str, TextChunk] = {}

        for result, normalized_score in zip(
            dense_results,
            dense_scores,
        ):
            chunk_id = result.chunk.chunk_id

            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0)

            scores[chunk_id] += (
                self.dense_weight * normalized_score
            )

        for result, normalized_score in zip(
            bm25_results,
            bm25_scores,
        ):
            chunk_id = result.chunk.chunk_id

            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0)

            scores[chunk_id] += (
                self.bm25_weight * normalized_score
            )

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            HybridSearchResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
            )
            for chunk_id in ranked_ids[:top_k]
        ]