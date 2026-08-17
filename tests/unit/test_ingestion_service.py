"""Ingestion orchestration tests without an external index."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_claim_verification.config import corpus_index_config_hash, load_corpus_config
from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.derivation import (
    prepare_derived_working_directory,
    resolve_derivation,
)
from rag_claim_verification.ingestion.loader import LoadedDocument, load_documents
from rag_claim_verification.ingestion.manifest import compute_corpus_hash
from rag_claim_verification.ingestion.service import (
    DERIVED_BASE_METADATA_NAME,
    INGESTION_METADATA_NAME,
    IngestionService,
    IngestionSummary,
    derived_from_summary,
    validate_ingested_index,
)
from rag_claim_verification.utils.files import write_json


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


def _base_summary(working: Path, *, content_hash: str) -> IngestionSummary:
    return IngestionSummary(
        corpus_id="clean",
        manifest_path="clean.jsonl",
        manifest_hash="clean-manifest",
        config_hash="clean-config",
        document_count=1,
        document_ids=["doc_1"],
        content_hashes={"doc_1": content_hash},
        working_directory=str(working),
        lightrag_version="1.5.4",
        tracking_id="base-track",
        completed_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def test_prepare_derived_working_directory_preserves_base_and_relabels_metadata(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    write_json(
        base / INGESTION_METADATA_NAME,
        _base_summary(base, content_hash="a" * 64).model_dump(mode="json"),
    )
    (base / "storage.json").write_text('{"base":true}\n', encoding="utf-8")
    target = tmp_path / "derived"

    prepare_derived_working_directory(
        base_working_directory=base,
        target_working_directory=target,
    )

    assert (base / INGESTION_METADATA_NAME).is_file()
    assert not (target / INGESTION_METADATA_NAME).exists()
    assert (target / DERIVED_BASE_METADATA_NAME).is_file()
    assert (target / "storage.json").read_text(encoding="utf-8") == '{"base":true}\n'


@pytest.mark.asyncio
async def test_derived_ingestion_only_sends_documents_beyond_base(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Base evidence", encoding="utf-8")
    (tmp_path / "noise.txt").write_text("Noise evidence", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "document_id": "doc_1",
                        "title": "Base",
                        "source": "Fixture",
                        "file_path": "doc.txt",
                    }
                ),
                json.dumps(
                    {
                        "document_id": "noise_1",
                        "title": "Noise",
                        "source": "Fixture",
                        "file_path": "noise.txt",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_documents(manifest)
    base_hash = loaded[0].content_hash
    working = tmp_path / "derived"
    working.mkdir()
    base_summary = _base_summary(working, content_hash=base_hash)
    base_metadata = working / DERIVED_BASE_METADATA_NAME
    write_json(base_metadata, base_summary.model_dump(mode="json"))
    provenance = derived_from_summary(base_summary, metadata_path=base_metadata)
    ingestor = RecordingIngestor()
    service = IngestionService(
        corpus_id="noisy",
        manifest_path=manifest,
        manifest_hash="noisy-manifest",
        config_hash="noisy-config",
        working_directory=working,
        lightrag_version="1.5.4",
        ingestor=ingestor,
        derived_from=provenance,
        prepared_derived_directory=True,
    )

    summary = await service.run()

    assert [item.metadata.document_id for item in ingestor.documents] == ["noise_1"]
    assert summary.document_ids == ["doc_1", "noise_1"]
    assert summary.derived_from == provenance
    validate_ingested_index(
        corpus_id="noisy",
        manifest_hash="noisy-manifest",
        config_hash="noisy-config",
        working_directory=working,
        lightrag_version="1.5.4",
        derived_from=provenance,
    )


@pytest.mark.asyncio
async def test_derived_ingestion_rejects_changed_base_content(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    working = tmp_path / "derived"
    working.mkdir()
    base_summary = _base_summary(working, content_hash="0" * 64)
    base_metadata = working / DERIVED_BASE_METADATA_NAME
    write_json(base_metadata, base_summary.model_dump(mode="json"))
    provenance = derived_from_summary(base_summary, metadata_path=base_metadata)
    service = IngestionService(
        corpus_id="noisy",
        manifest_path=manifest,
        manifest_hash="noisy-manifest",
        config_hash="noisy-config",
        working_directory=working,
        lightrag_version="1.5.4",
        ingestor=RecordingIngestor(),
        derived_from=provenance,
        prepared_derived_directory=True,
    )

    with pytest.raises(ManifestError, match="changed base content"):
        await service.run()


def test_resolve_derivation_validates_base_index_and_corpus_superset(tmp_path: Path) -> None:
    (tmp_path / "clean.txt").write_text("Clean evidence", encoding="utf-8")
    (tmp_path / "noise.txt").write_text("Noise evidence", encoding="utf-8")
    clean_manifest = tmp_path / "clean.jsonl"
    noisy_manifest = tmp_path / "noisy.jsonl"
    clean_record = {
        "document_id": "clean_1",
        "title": "Clean",
        "source": "Fixture",
        "file_path": "clean.txt",
    }
    noise_record = {
        "document_id": "noise_1",
        "title": "Noise",
        "source": "Fixture",
        "file_path": "noise.txt",
    }
    clean_manifest.write_text(json.dumps(clean_record) + "\n", encoding="utf-8")
    noisy_manifest.write_text(
        json.dumps(clean_record) + "\n" + json.dumps(noise_record) + "\n",
        encoding="utf-8",
    )
    retriever = {
        "type": "lightrag",
        "query_mode": "hybrid",
        "lightrag_llm": {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key_required": False,
            "model": "local-model",
        },
        "embedding": {
            "provider": "fastembed",
            "model": "fixture-embedding",
            "dimension": 2,
        },
    }
    base_config_path = tmp_path / "base.yaml"
    target_config_path = tmp_path / "target.yaml"
    write_json(
        base_config_path,
        {
            "corpus_id": "base",
            "manifest_path": "clean.jsonl",
            "retriever": {**retriever, "working_directory": "base-index"},
        },
    )
    write_json(
        target_config_path,
        {
            "corpus_id": "derived",
            "manifest_path": "noisy.jsonl",
            "clean_manifest_path": "clean.jsonl",
            "derived_from": {"corpus_config": "base.yaml"},
            "retriever": {**retriever, "working_directory": "derived-index"},
        },
    )
    base_config = load_corpus_config(base_config_path)
    base_working = base_config.retriever.working_directory
    assert base_working is not None
    base_working.mkdir()
    loaded = load_documents(clean_manifest)
    summary = IngestionSummary(
        corpus_id=base_config.corpus_id,
        manifest_path=str(clean_manifest),
        manifest_hash=compute_corpus_hash(clean_manifest),
        config_hash=corpus_index_config_hash(base_config),
        document_count=1,
        document_ids=["clean_1"],
        content_hashes={"clean_1": loaded[0].content_hash},
        working_directory=str(base_working),
        lightrag_version="1.5.4",
        completed_at=datetime(2026, 8, 17, tzinfo=UTC),
    )
    write_json(
        base_working / INGESTION_METADATA_NAME,
        summary.model_dump(mode="json"),
    )

    resolved = resolve_derivation(load_corpus_config(target_config_path))

    assert resolved is not None
    assert resolved.base_config.corpus_id == "base"
    assert resolved.provenance.document_ids == ["clean_1"]
