"""Manifest parsing and corpus loading tests."""

import json
from pathlib import Path

import pytest

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.loader import load_documents
from rag_claim_verification.ingestion.manifest import (
    parse_manifest,
    validate_noisy_superset,
)


def _record(document_id: str, file_path: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "title": f"Title {document_id}",
        "source": "Fixture",
        "file_path": file_path,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_manifest_parsing(tmp_path: Path) -> None:
    text_path = tmp_path / "doc.txt"
    text_path.write_text("Evidence", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [_record("doc_1", "doc.txt")])

    documents = parse_manifest(manifest)

    assert [item.document_id for item in documents] == ["doc_1"]


def test_manifest_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [_record("doc_1", "one.txt"), _record("doc_1", "two.txt")],
    )

    with pytest.raises(ManifestError, match="Duplicate document_id"):
        parse_manifest(manifest)


def test_loader_reports_all_missing_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [_record("doc_1", "one.txt"), _record("doc_2", "two.txt")],
    )

    with pytest.raises(ManifestError) as error:
        load_documents(manifest)

    assert "doc_1" in str(error.value)
    assert "doc_2" in str(error.value)


def test_loader_rejects_empty_document(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [_record("doc_1", "empty.txt")])

    with pytest.raises(ManifestError, match="empty document"):
        load_documents(manifest)


def test_noisy_manifest_must_include_clean_document_content(tmp_path: Path) -> None:
    (tmp_path / "clean.txt").write_text("clean", encoding="utf-8")
    (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
    clean_manifest = tmp_path / "clean.jsonl"
    noisy_manifest = tmp_path / "noisy.jsonl"
    _write_jsonl(clean_manifest, [_record("doc_1", "clean.txt")])
    _write_jsonl(noisy_manifest, [_record("doc_1", "changed.txt")])

    with pytest.raises(ManifestError, match="changed clean document content"):
        validate_noisy_superset(clean_manifest, noisy_manifest)
