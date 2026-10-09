from __future__ import annotations

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import RerankedResult, Reranker


class RerankedRetriever:
    """Retrieve candidates with hybrid search, then rerank them."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        retrieval_k: int = 5,
        top_k: int = 3,
    ) -> list[RerankedResult]:
        """Retrieve candidates with hybrid search and rerank them."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if retrieval_k <= 0:
            raise ValueError(
                "retrieval_k must be greater than 0."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        hybrid_results = self.hybrid_retriever.search(
            query=query,
            top_k=retrieval_k,
        )

        candidate_chunks = [
            result.chunk
            for result in hybrid_results
        ]

        return self.reranker.rerank(
            query=query,
            chunks=candidate_chunks,
            top_k=top_k,
        )