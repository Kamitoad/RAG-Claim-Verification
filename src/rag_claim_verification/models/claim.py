"""Claim and verdict models."""

from enum import StrEnum

from pydantic import Field, field_validator

from rag_claim_verification.models.base import StrictModel


class VerdictLabel(StrEnum):
    """Allowed three-way claim-verification labels."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


class Claim(StrictModel):
    """A pre-formulated atomic claim and optional benchmark ground truth."""

    claim_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    claim: str = Field(min_length=1)
    gold_label: VerdictLabel | None = None
    gold_document_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("gold_document_ids")
    @classmethod
    def validate_gold_document_ids(cls, values: list[str]) -> list[str]:
        """Reject blank or duplicate ground-truth document identifiers."""

        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("gold_document_ids must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("gold_document_ids must be unique")
        return normalized
