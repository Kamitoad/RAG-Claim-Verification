"""Structured model output and persisted prediction models."""

from enum import StrEnum
from typing import Literal

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


class CaseStatus(StrEnum):
    """Observable terminal state for one claim-condition evaluation case."""

    SUCCESS = "success"
    PARSE_ERROR = "parse_error"
    RETRIEVAL_ERROR = "retrieval_error"
    MODEL_ERROR = "model_error"
    COMPONENT_ERROR = "component_error"
    PIPELINE_ERROR = "pipeline_error"


class RetrievalStatus(StrEnum):
    """Whether retrieval was applicable, successful, empty, or failed."""

    NOT_APPLICABLE = "not_applicable"
    NOT_STARTED = "not_started"
    SUCCESS = "success"
    SUCCESS_EMPTY = "success_empty"
    ERROR = "error"


class ParseStatus(StrEnum):
    """Structured-output validation outcome, including the bounded repair path."""

    NOT_STARTED = "not_started"
    VALID_FIRST_PASS = "valid_first_pass"
    VALID_AFTER_REPAIR = "valid_after_repair"
    INVALID_AFTER_REPAIR = "invalid_after_repair"
    REPAIR_CALL_FAILED = "repair_call_failed"


class ModelCallMetadata(StrictModel):
    """Non-secret metadata for one initial or repair generation call."""

    purpose: Literal["initial", "repair"]
    provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    response_model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(ge=1)
    latency_ms: int = Field(ge=0)


class Prediction(StrictModel):
    """Persisted outcome for one claim under one experimental condition."""

    schema_version: Literal["2.0"] = "2.0"
    case_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    case_status: CaseStatus
    retrieval_status: RetrievalStatus
    parse_status: ParseStatus
    predicted_label: VerdictLabel | None
    reason: str | None
    evidence: list[Evidence] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    retrieval_latency_ms: int = Field(default=0, ge=0)
    generation_latency_ms: int = Field(default=0, ge=0)
    repair_latency_ms: int = Field(default=0, ge=0)
    model_calls: list[ModelCallMetadata] = Field(default_factory=list)
    raw_model_output: str | None = None
    repair_model_output: str | None = None
    initial_parse_error: str | None = None
    parse_error: str | None = None
    error: str | None = None
    error_stage: str | None = None
    error_type: str | None = None
    gold_label: VerdictLabel | None = None
    gold_document_ids: list[str] = Field(default_factory=list)
    retrieval_supports_document_ids: bool = False
    verification_mode: str = Field(pattern=r"^(rag|baseline)$")

    @model_validator(mode="after")
    def validate_outcome(self) -> "Prediction":
        """Ensure failed parsing never masquerades as a valid verdict."""

        if self.case_id != f"{self.condition}:{self.claim_id}":
            raise ValueError("case_id must equal '<condition>:<claim_id>'")
        if self.parse_error is not None and self.predicted_label is not None:
            raise ValueError("a parse-error prediction cannot contain a predicted label")
        if self.predicted_label is None and self.parse_error is None and self.error is None:
            raise ValueError("a prediction without a label must contain parse_error or error")
        if self.error is not None and (self.error_stage is None or self.error_type is None):
            raise ValueError("a technical error must contain error_stage and error_type")
        if self.error is None and (self.error_stage is not None or self.error_type is not None):
            raise ValueError("error_stage and error_type require a technical error")
        if self.case_status == CaseStatus.SUCCESS and self.predicted_label is None:
            raise ValueError("a successful case must contain a predicted label")
        if self.case_status == CaseStatus.PARSE_ERROR and self.parse_error is None:
            raise ValueError("parse_error case status requires parse_error details")
        if (
            self.case_status not in {CaseStatus.SUCCESS, CaseStatus.PARSE_ERROR}
            and self.error is None
        ):
            raise ValueError("a technical failure case status requires error details")
        return self
