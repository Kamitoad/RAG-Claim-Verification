"""Ingestion orchestration tests without an external index."""

import json
from pathlib import Path

import pytest

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.ingestion.service import IngestionService


class RecordingIngestor:
    def __init__(self) -> None:
        self.documents: list[LoadedDocument] = []

    async def ingest(self, documents: list[LoadedDocument]) -> str:
        self.documents = documents
        return "track-1"


def _manifest(tmp_path: Path) -> Path:
    (tmp_path / "doc.txt").write_text("Evidence", encoding="utf-8")
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "document_id": "doc_1",
                "title": "Title",
                "source": "Fixture",
                "file_path": "doc.txt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_ingestion_writes_reproducibility_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    ingestor = RecordingIngestor()
    working = tmp_path / "index"
    service = IngestionService(
        corpus_id="clean",
        manifest_path=manifest,
        manifest_hash="manifest-hash",
        config_hash="config-hash",
        working_directory=working,
        lightrag_version="1.5.4",
        ingestor=ingestor,
    )

    summary = await service.run()

    assert summary.document_ids == ["doc_1"]
    assert summary.tracking_id == "track-1"
    assert len(ingestor.documents) == 1
    assert (working / "ragcv_ingestion_metadata.json").is_file()


@pytest.mark.asyncio
async def test_ingestion_refuses_different_manifest_in_same_index(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    working = tmp_path / "index"
    first = IngestionService(
        corpus_id="clean",
        manifest_path=manifest,
        manifest_hash="first",
        config_hash="config",
        working_directory=working,
        lightrag_version="1.5.4",
        ingestor=RecordingIngestor(),
    )
    await first.run()
    second = IngestionService(
        corpus_id="clean",
        manifest_path=manifest,
        manifest_hash="second",
        config_hash="config",
        working_directory=working,
        lightrag_version="1.5.4",
        ingestor=RecordingIngestor(),
    )

    with pytest.raises(ManifestError, match="different corpus, manifest"):
        await second.run()
