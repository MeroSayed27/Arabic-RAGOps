from types import SimpleNamespace

import pytest

from evaluation.retrieval_metrics import (
    mean,
    recall_at_k,
    reciprocal_rank,
)


def make_result(chunk_id: str):
    """Create a small sample retrieval result."""
    return SimpleNamespace(
        chunk=SimpleNamespace(chunk_id=chunk_id)
    )


def test_recall_at_k_counts_expected_chunks_correctly():
    results = [
        make_result("chunk_a"),
        make_result("chunk_x"),
        make_result("chunk_b"),
    ]

    expected = ["chunk_a", "chunk_b"]

    assert recall_at_k(results, expected, k=1) == 0.5
    assert recall_at_k(results, expected, k=3) == 1.0


def test_recall_at_k_returns_zero_when_nothing_matches():
    results = [make_result("chunk_x")]

    assert recall_at_k(results, ["chunk_a"], k=1) == 0.0


def test_reciprocal_rank_rewards_earlier_results():
    results = [
        make_result("chunk_x"),
        make_result("chunk_a"),
    ]

    assert reciprocal_rank(results, ["chunk_a"]) == 0.5


def test_reciprocal_rank_returns_zero_when_nothing_matches():
    results = [make_result("chunk_x")]

    assert reciprocal_rank(results, ["chunk_a"]) == 0.0


def test_mean_calculates_average():
    assert mean([1.0, 0.5, 0.0]) == 0.5


def test_recall_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(
            [make_result("chunk_a")],
            ["chunk_a"],
            k=0,
        )