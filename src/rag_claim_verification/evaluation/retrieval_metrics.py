"""Retrieval metrics computed only from concrete ranked document IDs."""

from collections.abc import Sequence
from typing import Any


def evidence_recall_at_k(
    gold_document_ids: Sequence[set[str]],
    retrieved_document_ids: Sequence[list[str]],
    k: int,
) -> float:
    """Average the fraction of each claim's gold documents present in the top k."""

    if k <= 0:
        raise ValueError("k must be positive")
    if len(gold_document_ids) != len(retrieved_document_ids):
        raise ValueError("gold and retrieved sequences must have equal length")
    if not gold_document_ids:
        raise ValueError("at least one retrieval case is required")
    if any(not item for item in gold_document_ids):
        raise ValueError("each retrieval case must contain at least one gold document ID")
    recalls = [
        len(gold & set(retrieved[:k])) / len(gold)
        for gold, retrieved in zip(gold_document_ids, retrieved_document_ids, strict=True)
    ]
    return sum(recalls) / len(recalls)


def hit_rate_at_k(
    gold_document_ids: Sequence[set[str]],
    retrieved_document_ids: Sequence[list[str]],
    k: int,
) -> float:
    """Return the fraction of cases with at least one gold document in the top k."""

    if k <= 0:
        raise ValueError("k must be positive")
    if len(gold_document_ids) != len(retrieved_document_ids):
        raise ValueError("gold and retrieved sequences must have equal length")
    if not gold_document_ids:
        raise ValueError("at least one retrieval case is required")
    if any(not item for item in gold_document_ids):
        raise ValueError("each retrieval case must contain at least one gold document ID")
    hits = [
        bool(gold & set(retrieved[:k]))
        for gold, retrieved in zip(gold_document_ids, retrieved_document_ids, strict=True)
    ]
    return sum(hits) / len(hits)


def mean_reciprocal_rank(
    gold_document_ids: Sequence[set[str]], retrieved_document_ids: Sequence[list[str]]
) -> float:
    """Compute reciprocal rank of the first relevant retrieved document."""

    if len(gold_document_ids) != len(retrieved_document_ids):
        raise ValueError("gold and retrieved sequences must have equal length")
    if not gold_document_ids:
        raise ValueError("at least one retrieval case is required")
    reciprocal_ranks: list[float] = []
    for gold, retrieved in zip(gold_document_ids, retrieved_document_ids, strict=True):
        if not gold:
            raise ValueError("each retrieval case must contain at least one gold document ID")
        first_rank = next(
            (rank for rank, document_id in enumerate(retrieved, start=1) if document_id in gold),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(
    gold_document_ids: Sequence[set[str]], retrieved_document_ids: Sequence[list[str]]
) -> dict[str, Any]:
    """Compute the MVP retrieval metric set for eligible claims."""

    return {
        "eligible_claim_count": len(gold_document_ids),
        "evidence_recall_at_1": evidence_recall_at_k(gold_document_ids, retrieved_document_ids, 1),
        "evidence_recall_at_3": evidence_recall_at_k(gold_document_ids, retrieved_document_ids, 3),
        "evidence_recall_at_5": evidence_recall_at_k(gold_document_ids, retrieved_document_ids, 5),
        "hit_rate_at_1": hit_rate_at_k(gold_document_ids, retrieved_document_ids, 1),
        "hit_rate_at_3": hit_rate_at_k(gold_document_ids, retrieved_document_ids, 3),
        "hit_rate_at_5": hit_rate_at_k(gold_document_ids, retrieved_document_ids, 5),
        "mean_reciprocal_rank": mean_reciprocal_rank(gold_document_ids, retrieved_document_ids),
    }
