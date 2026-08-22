"""Reproducible ingestion orchestration independent of LightRAG internals."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field, ValidationError, model_validator

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.loader import LoadedDocument, load_documents
from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.utils.files import read_yaml, write_json
from rag_claim_verification.utils.hashing import sha256_file

INGESTION_METADATA_NAME = "ragcv_ingestion_metadata.json"
DERIVED_BASE_METADATA_NAME = "ragcv_derived_from_metadata.json"


class CorpusIngestor(Protocol):
    """Small interface implemented by an index adapter."""

    async def ingest(self, documents: list[LoadedDocument]) -> str | None:
        """Insert an already validated document batch and return an optional tracking ID."""


class IndexIdentity(StrictModel):
    """Stable identity fields that determine whether an index matches a corpus config."""

    corpus_id: str
    manifest_hash: str
    config_hash: str
    lightrag_version: str


class DerivedFromIndex(StrictModel):
    """Exact immutable base-index snapshot recorded by a derived index."""

    identity: IndexIdentity
    ingestion_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_ids: list[str] = Field(min_length=1)
    content_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_document_hashes(self) -> "DerivedFromIndex":
        """Require one content hash for every unique base document."""

        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("derived base document_ids must be unique")
        if set(self.document_ids) != set(self.content_hashes):
            raise ValueError("derived base document_ids and content_hashes must match")
        return self


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
    derived_from: DerivedFromIndex | None = None
    completed_at: datetime

    @property
    def identity(self) -> IndexIdentity:
        """Return the four stable fields used by index validation."""

        return IndexIdentity(
            corpus_id=self.corpus_id,
            manifest_hash=self.manifest_hash,
            config_hash=self.config_hash,
            lightrag_version=self.lightrag_version,
        )


def read_ingestion_summary(metadata_path: Path) -> IngestionSummary:
    """Strictly parse one persisted RAGCV ingestion summary."""

    try:
        return IngestionSummary.model_validate(read_yaml(metadata_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ManifestError(f"Invalid ingestion metadata {metadata_path}: {exc}") from exc


def derived_from_summary(
    summary: IngestionSummary,
    *,
    metadata_path: Path,
) -> DerivedFromIndex:
    """Capture the exact identity and content of a validated base-index summary."""

    return DerivedFromIndex(
        identity=summary.identity,
        ingestion_metadata_sha256=sha256_file(metadata_path),
        document_ids=summary.document_ids,
        content_hashes=summary.content_hashes,
    )


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
        derived_from: DerivedFromIndex | None = None,
        prepared_derived_directory: bool = False,
    ) -> None:
        self._corpus_id = corpus_id
        self._manifest_path = manifest_path
        self._manifest_hash = manifest_hash
        self._config_hash = config_hash
        self._working_directory = working_directory
        self._lightrag_version = lightrag_version
        self._ingestor = ingestor
        self._derived_from = derived_from
        self._prepared_derived_directory = prepared_derived_directory
        if prepared_derived_directory and derived_from is None:
            raise ValueError("prepared_derived_directory requires derived_from metadata")

    async def run(self) -> IngestionSummary:
        """Run all-or-nothing validation before any external indexing call."""

        documents = load_documents(self._manifest_path)
        metadata_path = self._working_directory / INGESTION_METADATA_NAME
        if metadata_path.exists():
            existing = read_ingestion_summary(metadata_path)
            expected = IndexIdentity(
                corpus_id=self._corpus_id,
                manifest_hash=self._manifest_hash,
                config_hash=self._config_hash,
                lightrag_version=self._lightrag_version,
            )
            if existing.identity != expected or existing.derived_from != self._derived_from:
                raise ManifestError(
                    "Working directory already belongs to a different corpus, manifest, "
                    "index configuration, LightRAG version, or derivation base: "
                    f"{self._working_directory}"
                )
        elif self._prepared_derived_directory:
            base_metadata_path = self._working_directory / DERIVED_BASE_METADATA_NAME
            if not base_metadata_path.is_file():
                raise ManifestError(
                    "Prepared derived index is missing its immutable base metadata: "
                    f"{base_metadata_path}"
                )
            base_summary = read_ingestion_summary(base_metadata_path)
            observed_base = derived_from_summary(
                base_summary,
                metadata_path=base_metadata_path,
            )
            if observed_base != self._derived_from:
                raise ManifestError(
                    "Prepared derived index does not match the declared derivation base: "
                    f"{self._working_directory}"
                )
        elif self._working_directory.is_dir() and any(self._working_directory.iterdir()):
            raise ManifestError(
                "Working directory is non-empty but has no RAGCV ingestion metadata; "
                f"refusing to mix index data: {self._working_directory}"
            )

        self._working_directory.mkdir(parents=True, exist_ok=True)
        documents_to_ingest = documents
        if self._derived_from is not None:
            documents_by_id = {item.metadata.document_id: item for item in documents}
            missing = sorted(set(self._derived_from.document_ids) - set(documents_by_id))
            changed = sorted(
                document_id
                for document_id, expected_hash in self._derived_from.content_hashes.items()
                if document_id in documents_by_id
                and documents_by_id[document_id].content_hash != expected_hash
            )
            if missing or changed:
                details: list[str] = []
                if missing:
                    details.append("missing base IDs: " + ", ".join(missing))
                if changed:
                    details.append("changed base content: " + ", ".join(changed))
                raise ManifestError(
                    "Derived corpus does not preserve its base index; " + "; ".join(details)
                )
            base_ids = set(self._derived_from.document_ids)
            documents_to_ingest = [
                document for document in documents if document.metadata.document_id not in base_ids
            ]
            if not documents_to_ingest:
                raise ManifestError("Derived corpus contains no documents beyond its base index")

        tracking_id = await self._ingestor.ingest(documents_to_ingest)
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
            derived_from=self._derived_from,
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
    derived_from: DerivedFromIndex | None = None,
) -> None:
    """Verify that an existing LightRAG index matches the resolved corpus configuration."""

    metadata_path = working_directory / INGESTION_METADATA_NAME
    if not metadata_path.is_file():
        raise ManifestError(
            f"No ingestion metadata found for corpus {corpus_id!r}; run ingest first: "
            f"{metadata_path}"
        )
    existing = read_ingestion_summary(metadata_path)
    expected = IndexIdentity(
        corpus_id=corpus_id,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
        lightrag_version=lightrag_version,
    )
    if existing.identity != expected or existing.derived_from != derived_from:
        raise ManifestError(
            "Ingested index metadata does not match the current corpus, documents, "
            f"configuration, LightRAG version, or derivation base: {working_directory}"
        )
