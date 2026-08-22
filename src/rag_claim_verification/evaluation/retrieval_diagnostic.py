"""Descriptive retrieval diagnostics without claim verification or invented scores."""

from typing import Literal

from pydantic import Field, model_validator

from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.models.claim import Claim
from rag_claim_verification.models.evidence import Evidence

QueryMode = Literal["local", "global", "hybrid", "naive", "mix"]


class RetrievalDiagnosticCase(StrictModel):
    """One claim's ordered retrieval observation under one query mode."""

    mode: QueryMode
    claim_id: str
    claim: str
    gold_document_ids: list[str]
    retrieved_document_ids: list[str | None]
    noise_document_ids: list[str]
    unmapped_evidence_count: int = Field(ge=0)
    gold_best_rank: int | None = Field(default=None, ge=1)
    retrieval_latency_ms: int = Field(ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_error_pair(self) -> "RetrievalDiagnosticCase":
        """Require diagnostic errors to remain explicit and internally consistent."""

        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("error_type and error_message must either both be set or both be null")
        return self


class RetrievalModeSummary(StrictModel):
    """Small descriptive summary for one retrieval mode."""

    mode: QueryMode
    case_count: int = Field(ge=0)
    successful_case_count: int = Field(ge=0)
    error_case_count: int = Field(ge=0)
    noise_exposed_case_count: int = Field(ge=0)
    noise_hit_count: int = Field(ge=0)
    unmapped_evidence_count: int = Field(ge=0)
    gold_eligible_case_count: int = Field(ge=0)
    gold_retrieved_case_count: int = Field(ge=0)
    gold_rank_1_case_count: int = Field(ge=0)


def build_diagnostic_case(
    *,
    mode: QueryMode,
    claim: Claim,
    evidence: list[Evidence],
    noise_ids: set[str],
    retrieval_latency_ms: int,
    error: Exception | None = None,
) -> RetrievalDiagnosticCase:
    """Convert provider evidence into an honest document-level diagnostic observation."""

    retrieved_ids = [item.document_id for item in evidence]
    gold_ids = set(claim.gold_document_ids)
    gold_ranks = [
        item.rank
        for item in evidence
        if item.document_id is not None and item.document_id in gold_ids
    ]
    return RetrievalDiagnosticCase(
        mode=mode,
        claim_id=claim.claim_id,
        claim=claim.claim,
        gold_document_ids=claim.gold_document_ids,
        retrieved_document_ids=retrieved_ids,
        noise_document_ids=[
            document_id
            for document_id in retrieved_ids
            if document_id is not None and document_id in noise_ids
        ],
        unmapped_evidence_count=sum(document_id is None for document_id in retrieved_ids),
        gold_best_rank=min(gold_ranks) if gold_ranks else None,
        retrieval_latency_ms=retrieval_latency_ms,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
    )


def summarize_mode(
    mode: QueryMode,
    cases: list[RetrievalDiagnosticCase],
) -> RetrievalModeSummary:
    """Aggregate only directly observable retrieval counts for one mode."""

    mode_cases = [case for case in cases if case.mode == mode]
    return RetrievalModeSummary(
        mode=mode,
        case_count=len(mode_cases),
        successful_case_count=sum(case.error_type is None for case in mode_cases),
        error_case_count=sum(case.error_type is not None for case in mode_cases),
        noise_exposed_case_count=sum(bool(case.noise_document_ids) for case in mode_cases),
        noise_hit_count=sum(len(case.noise_document_ids) for case in mode_cases),
        unmapped_evidence_count=sum(case.unmapped_evidence_count for case in mode_cases),
        gold_eligible_case_count=sum(bool(case.gold_document_ids) for case in mode_cases),
        gold_retrieved_case_count=sum(
            bool(case.gold_document_ids) and case.gold_best_rank is not None for case in mode_cases
        ),
        gold_rank_1_case_count=sum(case.gold_best_rank == 1 for case in mode_cases),
    )
