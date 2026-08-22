"""Run completeness and separated evaluation-metric tests."""

import pytest

from rag_claim_verification.evaluation.reporting import (
    evaluate_predictions,
    validate_case_completeness,
)
from rag_claim_verification.models.claim import VerdictLabel
from rag_claim_verification.models.prediction import (
    CaseStatus,
    ParseStatus,
    Prediction,
    RetrievalStatus,
)
from rag_claim_verification.models.run import CaseManifestRecord


def _case(condition: str = "baseline") -> CaseManifestRecord:
    return CaseManifestRecord(
        case_id=f"{condition}:claim_1",
        sequence=1,
        claim_id="claim_1",
        claim="A fixture claim",
        gold_label=VerdictLabel.SUPPORTED,
        gold_document_ids=["doc_1"],
        condition=condition,
        verification_mode="baseline" if condition == "baseline" else "rag",
    )


def _prediction(condition: str = "baseline") -> Prediction:
    return Prediction(
        case_id=f"{condition}:claim_1",
        claim_id="claim_1",
        claim="A fixture claim",
        condition=condition,
        case_status=CaseStatus.SUCCESS,
        retrieval_status=(
            RetrievalStatus.NOT_APPLICABLE
            if condition == "baseline"
            else RetrievalStatus.SUCCESS_EMPTY
        ),
        parse_status=ParseStatus.VALID_FIRST_PASS,
        predicted_label=VerdictLabel.SUPPORTED,
        reason="Fixture reason",
        latency_ms=2,
        generation_latency_ms=1,
        gold_label=VerdictLabel.SUPPORTED,
        gold_document_ids=["doc_1"],
        verification_mode="baseline" if condition == "baseline" else "rag",
    )


def test_completeness_rejects_missing_case_result() -> None:
    with pytest.raises(ValueError, match="missing=baseline:claim_1"):
        validate_case_completeness([_case()], [])


def test_completeness_rejects_duplicate_case_result() -> None:
    prediction = _prediction()
    with pytest.raises(ValueError, match="duplicate case IDs"):
        validate_case_completeness([_case()], [prediction, prediction])


def test_evaluation_separates_completeness_and_condition_metrics() -> None:
    metrics = evaluate_predictions([_prediction()], [_case()])

    assert metrics["completeness"]["complete"] is True
    condition = metrics["conditions"]["baseline"]
    assert condition["classification"]["accuracy"] == 1.0
    assert condition["retrieval"] is None
    assert condition["structured_output"]["first_pass_valid_rate"] == 1.0
    assert condition["technical_errors"]["count"] == 0
