"""Retrieval metric tests."""

import pytest

from rag_claim_verification.evaluation.retrieval_metrics import (
    evidence_recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)


def test_evidence_recall_at_k_uses_fraction_of_gold_documents() -> None:
    gold = [{"a", "b"}, {"c"}]
    retrieved = [["a", "x", "b"], ["x", "c"]]

    assert evidence_recall_at_k(gold, retrieved, 1) == pytest.approx(0.25)
    assert evidence_recall_at_k(gold, retrieved, 3) == pytest.approx(1.0)


def test_mean_reciprocal_rank() -> None:
    gold = [{"a"}, {"b"}, {"missing"}]
    retrieved = [["x", "a"], ["b"], ["x"]]

    assert mean_reciprocal_rank(gold, retrieved) == pytest.approx(0.5)


def test_hit_rate_at_k_counts_cases_with_any_gold_document() -> None:
    gold = [{"a", "b"}, {"c"}]
    retrieved = [["x", "a"], ["c"]]

    assert hit_rate_at_k(gold, retrieved, 1) == pytest.approx(0.5)
    assert hit_rate_at_k(gold, retrieved, 2) == pytest.approx(1.0)
