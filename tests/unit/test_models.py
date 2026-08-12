"""Domain-model validation tests."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.document import Document
from rag_claim_verification.models.prediction import (
    CaseStatus,
    ParseStatus,
    Prediction,
    RetrievalStatus,
)


def test_label_validation_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        Claim(claim_id="claim_1", claim="A claim", gold_label="MAYBE")  # type: ignore[arg-type]


def test_document_keeps_publication_and_event_dates_separate() -> None:
    document = Document(
        document_id="doc_1",
        title="Retrospective",
        source="Fixture",
        file_path=Path("doc.txt"),
        publication_date=date(2024, 1, 1),
        event_date=date(1994, 11, 13),
    )

    assert document.publication_date == date(2024, 1, 1)
    assert document.event_date == date(1994, 11, 13)


@pytest.mark.parametrize("field", ["document_id", "title", "source"])
def test_document_rejects_blank_required_fields(field: str) -> None:
    values = {
        "document_id": "doc_1",
        "title": "Title",
        "source": "Source",
        "file_path": Path("doc.txt"),
    }
    values[field] = " "
    with pytest.raises(ValidationError):
        Document.model_validate(values)


def test_claim_rejects_duplicate_gold_document_ids() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        Claim(
            claim_id="claim_1",
            claim="A claim",
            gold_label=VerdictLabel.SUPPORTED,
            gold_document_ids=["doc_1", "doc_1"],
        )


def test_parse_error_prediction_cannot_have_label() -> None:
    with pytest.raises(ValidationError, match="parse-error"):
        Prediction(
            case_id="test:claim_1",
            claim_id="claim_1",
            claim="A claim",
            condition="test",
            case_status=CaseStatus.PARSE_ERROR,
            retrieval_status=RetrievalStatus.SUCCESS_EMPTY,
            parse_status=ParseStatus.INVALID_AFTER_REPAIR,
            predicted_label=VerdictLabel.SUPPORTED,
            reason="Reason",
            latency_ms=1,
            parse_error="bad JSON",
            verification_mode="rag",
        )
