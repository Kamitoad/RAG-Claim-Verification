"""Benchmark run metadata models."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from rag_claim_verification.models.base import StrictModel


class RunStatus(StrEnum):
    """Lifecycle state written to run metadata."""

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class ConditionMetadata(StrictModel):
    """Resolved, non-secret settings for one benchmark condition."""

    condition_id: str
    mode: str
    model_name: str
    model_endpoint: str
    temperature: float
    retriever_type: str
    top_k: int | None = None
    corpus_id: str | None = None
    manifest_hash: str | None = None


class RunMetadata(StrictModel):
    """Environment and input provenance for a benchmark run."""

    run_id: str
    experiment_name: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    git_commit: str | None = None
    python_version: str
    platform: str
    package_versions: dict[str, str | None]
    config_file: str
    config_hash: str
    claims_file: str
    claims_hash: str
    prompt_hash: str
    prompt_version: str
    conditions: list[ConditionMetadata]
    prediction_count: int = Field(default=0, ge=0)
    successful_prediction_count: int = Field(default=0, ge=0)
    failed_prediction_count: int = Field(default=0, ge=0)
