"""Reproducible ingestion orchestration independent of LightRAG internals."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.loader import LoadedDocument, load_documents
from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.utils.files import read_yaml, write_json


class CorpusIngestor(Protocol):
    """Small interface implemented by an index adapter."""

    async def ingest(self, documents: list[LoadedDocument]) -> str | None:
        """Insert an already validated document batch and return an optional tracking ID."""


class IngestionSummary(StrictModel):
    """Persisted summary tying an index to exact inputs and configuration."""

    corpus_id: str
    manifest_path: str
    manifest_hash: str
    config_hash: str
    document_count: int = Field(ge=0)
    document_ids: list[str]
    content_hashes: dict[str, str]
    working_directory: str
    lightrag_version: str
    tracking_id: str | None = None
    completed_at: datetime


class IngestionService:
    """Validate inputs, protect index identity, and invoke the configured ingestor."""

    def __init__(
        self,
        *,
        corpus_id: str,
        manifest_path: Path,
        manifest_hash: str,
        config_hash: str,
        working_directory: Path,
        lightrag_version: str,
        ingestor: CorpusIngestor,
    ) -> None:
        self._corpus_id = corpus_id
        self._manifest_path = manifest_path
        self._manifest_hash = manifest_hash
        self._config_hash = config_hash
        self._working_directory = working_directory
        self._lightrag_version = lightrag_version
        self._ingestor = ingestor

    async def run(self) -> IngestionSummary:
        """Run all-or-nothing validation before any external indexing call."""

        documents = load_documents(self._manifest_path)
        metadata_path = self._working_directory / "ragcv_ingestion_metadata.json"
        if metadata_path.exists():
            # JSON is also valid YAML, which keeps the safe structured reader centralized.
            existing = read_yaml(metadata_path)
            identity = (
                existing.get("corpus_id"),
                existing.get("manifest_hash"),
                existing.get("config_hash"),
                existing.get("lightrag_version"),
            )
            expected = (
                self._corpus_id,
                self._manifest_hash,
                self._config_hash,
                self._lightrag_version,
            )
            if identity != expected:
                raise ManifestError(
                    "Working directory already belongs to a different corpus, manifest, "
                    "index configuration, or LightRAG version: "
                    f"{self._working_directory}"
                )
        elif self._working_directory.is_dir() and any(self._working_directory.iterdir()):
            raise ManifestError(
                "Working directory is non-empty but has no RAGCV ingestion metadata; "
                f"refusing to mix index data: {self._working_directory}"
            )

        self._working_directory.mkdir(parents=True, exist_ok=True)
        tracking_id = await self._ingestor.ingest(documents)
        summary = IngestionSummary(
            corpus_id=self._corpus_id,
            manifest_path=str(self._manifest_path),
            manifest_hash=self._manifest_hash,
            config_hash=self._config_hash,
            document_count=len(documents),
            document_ids=[item.metadata.document_id for item in documents],
            content_hashes={item.metadata.document_id: item.content_hash for item in documents},
            working_directory=str(self._working_directory),
            lightrag_version=self._lightrag_version,
            tracking_id=tracking_id,
            completed_at=datetime.now(UTC),
        )
        write_json(metadata_path, summary.model_dump(mode="json"))
        return summary


def validate_ingested_index(
    *,
    corpus_id: str,
    manifest_hash: str,
    config_hash: str,
    working_directory: Path,
    lightrag_version: str,
) -> None:
    """Verify that an existing LightRAG index matches the resolved corpus configuration."""

    metadata_path = working_directory / "ragcv_ingestion_metadata.json"
    if not metadata_path.is_file():
        raise ManifestError(
            f"No ingestion metadata found for corpus {corpus_id!r}; run ingest first: "
            f"{metadata_path}"
        )
    existing = read_yaml(metadata_path)
    actual = (
        existing.get("corpus_id"),
        existing.get("manifest_hash"),
        existing.get("config_hash"),
        existing.get("lightrag_version"),
    )
    expected = (corpus_id, manifest_hash, config_hash, lightrag_version)
    if actual != expected:
        raise ManifestError(
            "Ingested index metadata does not match the current corpus, documents, "
            f"configuration, or LightRAG version: {working_directory}"
        )
