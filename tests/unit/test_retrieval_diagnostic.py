"""Retrieval-only diagnostic tests without LightRAG or model calls."""

from rag_claim_verification.evaluation.retrieval_diagnostic import (
    build_diagnostic_case,
    summarize_mode,
)
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence


def _claim(*, with_gold: bool = True) -> Claim:
    return Claim(
        claim_id="claim_1",
        claim="Driver Alpha won the race.",
        gold_label=VerdictLabel.SUPPORTED,
        gold_document_ids=["clean_1"] if with_gold else [],
    )


def test_diagnostic_records_noise_gold_rank_and_unmapped_evidence() -> None:
    case = build_diagnostic_case(
        mode="naive",
        claim=_claim(),
        evidence=[
            Evidence(document_id="noise_1", text="Noise", rank=1),
            Evidence(document_id="clean_1", text="Gold", rank=2),
            Evidence(document_id=None, text="Unmapped", rank=3),
        ],
        noise_ids={"noise_1"},
        retrieval_latency_ms=12,
    )

    assert case.retrieved_document_ids == ["noise_1", "clean_1", None]
    assert case.noise_document_ids == ["noise_1"]
    assert case.gold_best_rank == 2
    assert case.unmapped_evidence_count == 1

    summary = summarize_mode("naive", [case])
    assert summary.noise_exposed_case_count == 1
    assert summary.noise_hit_count == 1
    assert summary.gold_retrieved_case_count == 1
    assert summary.gold_rank_1_case_count == 0


def test_diagnostic_keeps_provider_error_distinct_from_empty_retrieval() -> None:
    error_case = build_diagnostic_case(
        mode="mix",
        claim=_claim(with_gold=False),
        evidence=[],
        noise_ids={"noise_1"},
        retrieval_latency_ms=3,
        error=RuntimeError("provider failed"),
    )
    empty_case = build_diagnostic_case(
        mode="mix",
        claim=_claim(with_gold=False).model_copy(update={"claim_id": "claim_2"}),
        evidence=[],
        noise_ids={"noise_1"},
        retrieval_latency_ms=2,
    )

    summary = summarize_mode("mix", [error_case, empty_case])

    assert error_case.error_type == "RuntimeError"
    assert error_case.error_message == "provider failed"
    assert summary.case_count == 2
    assert summary.error_case_count == 1
    assert summary.successful_case_count == 1
