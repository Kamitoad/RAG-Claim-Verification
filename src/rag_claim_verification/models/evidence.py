"""Retrieved evidence model."""

import math
from datetime import date

from pydantic import Field, field_validator

from rag_claim_verification.models.base import StrictModel


class Evidence(StrictModel):
    """One ranked evidence passage returned by a retriever."""

    document_id: str | None = None
    text: str = Field(min_length=1)
    rank: int = Field(ge=1)
    retrieval_score: float | None = None
    source: str | None = None
    publication_date: date | None = None
    file_path: str | None = None
    chunk_id: str | None = None

    @field_validator("retrieval_score")
    @classmethod
    def validate_score(cls, value: float | None) -> float | None:
        """Reject non-finite scores without assuming a provider-specific range."""

        if value is not None and not math.isfinite(value):
            raise ValueError("retrieval_score must be finite")
        return value
