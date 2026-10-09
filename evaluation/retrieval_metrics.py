from __future__ import annotations

from typing import Iterable, Protocol

from app.chunking.text_chunker import TextChunk


class RetrievalResult(Protocol):
    """Common structure for retrieval results."""

    chunk: TextChunk


def recall_at_k(
    results: list[RetrievalResult],
    expected_chunk_ids: Iterable[str],
    k: int,
) -> float:
    """Calculate the proportion of expected chunks found in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    expected = set(expected_chunk_ids)

    if not expected:
        return 0.0

    retrieved_ids = {
        result.chunk.chunk_id
        for result in results[:k]
    }

    relevant_found = len(expected.intersection(retrieved_ids))

    return relevant_found / len(expected)


def reciprocal_rank(
    results: list[RetrievalResult],
    expected_chunk_ids: Iterable[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""
    expected = set(expected_chunk_ids)

    if not expected:
        return 0.0

    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in expected:
            return 1.0 / rank

    return 0.0


def mean(values: list[float]) -> float:
    """Calculate the arithmetic mean."""
    if not values:
        return 0.0

    return sum(values) / len(values)