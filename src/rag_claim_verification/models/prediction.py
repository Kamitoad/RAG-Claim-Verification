"""Structured model output and persisted prediction models."""

from pydantic import Field, field_validator, model_validator

from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.models.claim import VerdictLabel
from rag_claim_verification.models.evidence import Evidence


class VerificationOutput(StrictModel):
    """Strict JSON contract requested from the verification model."""

    label: VerdictLabel
    reason: str = Field(min_length=1, max_length=2000)
    cited_document_ids: list[str] = Field(default_factory=list)

    @field_validator("cited_document_ids")
    @classmethod
    def validate_citations(cls, values: list[str]) -> list[str]:
        """Reject blank and duplicate citations."""

        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("cited_document_ids must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("cited_document_ids must be unique")
        return normalized


class Prediction(StrictModel):
    """Persisted outcome for one claim under one experimental condition."""

    claim_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    predicted_label: VerdictLabel | None
    reason: str | None
    evidence: list[Evidence] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    retrieval_latency_ms: int = Field(default=0, ge=0)
    raw_model_output: str | None = None
    repair_model_output: str | None = None
    parse_error: str | None = None
    error: str | None = None
    gold_label: VerdictLabel | None = None
    gold_document_ids: list[str] = Field(default_factory=list)
    retrieval_supports_document_ids: bool = False
    verification_mode: str = Field(pattern=r"^(rag|baseline)$")

    @model_validator(mode="after")
    def validate_outcome(self) -> "Prediction":
        """Ensure failed parsing never masquerades as a valid verdict."""

        if self.parse_error is not None and self.predicted_label is not None:
            raise ValueError("a parse-error prediction cannot contain a predicted label")
        if self.predicted_label is None and self.parse_error is None and self.error is None:
            raise ValueError("a prediction without a label must contain parse_error or error")
        return self
