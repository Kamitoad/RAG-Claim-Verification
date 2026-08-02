"""Curated document metadata model."""

from datetime import date
from pathlib import Path

from pydantic import Field, field_validator

from rag_claim_verification.models.base import StrictModel


class Document(StrictModel):
    """Metadata for one locally stored corpus document.

    Publication and event dates are intentionally separate: a retrospective article
    may describe an event that happened many years before publication.
    """

    document_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    file_path: Path
    publication_date: date | None = None
    event_date: date | None = None
    topic: str | None = None
    language: str | None = Field(default=None, min_length=2, max_length=16)
    corpus_tags: list[str] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: Path) -> Path:
        """Require a file path instead of an empty current-directory path."""

        if str(value).strip() in {"", "."}:
            raise ValueError("file_path must identify a document file")
        return value

    @field_validator("corpus_tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        """Normalize tags while preserving their declared order."""

        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("corpus_tags must not contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("corpus_tags must be unique")
        return normalized
