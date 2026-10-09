from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.chunking.text_chunker import TextChunk


@dataclass(frozen=True)
class BM25SearchResult:
    """A BM25 search result for one chunk."""

    chunk: TextChunk
    score: float


class BM25Index:
    """Keyword-based retrieval index using BM25."""

    def __init__(self) -> None:
        self._chunks: list[TextChunk] = []
        self._tokenized_chunks: list[list[str]] = []
        self._index: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Create a simple multilingual token list."""
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Build the BM25 index from chunks."""
        if not chunks:
            return

        self._chunks.extend(chunks)

        tokenized = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self._tokenized_chunks.extend(tokenized)
        self._index = BM25Okapi(self._tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[BM25SearchResult]:
        """Return the most relevant chunks using BM25."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self._index is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            BM25SearchResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
            )
            for index in ranked_indices[:top_k]
        ]

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)