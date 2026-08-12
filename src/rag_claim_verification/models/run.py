"""Benchmark run metadata models."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.models.claim import VerdictLabel


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
    provider: str
    model_name: str
    model_endpoint: str
    temperature: float
    seed: int | None = None
    timeout_seconds: float
    max_retries: int
    request_json_object: bool
    retriever_type: str
    top_k: int | None = None
    corpus_id: str | None = None
    corpus_config_file: str | None = None
    corpus_config_hash: str | None = None
    manifest_file: str | None = None
    manifest_file_hash: str | None = None
    corpus_hash: str | None = None


class RunMetadata(StrictModel):
    """Environment and input provenance for a benchmark run."""

    schema_version: Literal["2.0"] = "2.0"
    run_id: str
    run_group_id: str
    repetition_index: int = Field(default=1, ge=1)
    experiment_name: str
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    git_commit: str | None = None
    source_tree_hash: str | None = None
    python_version: str
    platform: str
    package_versions: dict[str, str | None]
    config_file: str
    config_hash: str
    resolved_config_hash: str
    claims_file: str
    claims_hash: str
    claims_snapshot_hash: str
    prompt_hash: str
    prompt_version: str
    prompt_hashes: dict[str, str]
    case_manifest_hash: str
    git_dirty: bool | None = None
    dependency_lock_file: str | None = None
    dependency_lock_hash: str | None = None
    timing_definitions: dict[str, str]
    conditions: list[ConditionMetadata]
    expected_prediction_count: int = Field(ge=0)
    prediction_count: int = Field(default=0, ge=0)
    predictions_hash: str | None = None
    successful_prediction_count: int = Field(default=0, ge=0)
    failed_prediction_count: int = Field(default=0, ge=0)
    missing_prediction_count: int = Field(default=0, ge=0)
    benchmark_duration_ms: int | None = Field(default=None, ge=0)
    run_error: str | None = None
    run_error_stage: str | None = None
    run_error_type: str | None = None


class CaseManifestRecord(StrictModel):
    """One planned claim-condition pair written before external calls begin."""

    schema_version: Literal["2.0"] = "2.0"
    case_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    claim_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    gold_label: VerdictLabel
    gold_document_ids: list[str]
    condition: str = Field(min_length=1)
    verification_mode: Literal["baseline", "rag"]
