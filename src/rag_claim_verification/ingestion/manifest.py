"""Document manifest parsing and cross-corpus validation."""

from pathlib import Path

from pydantic import ValidationError

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.models.document import Document
from rag_claim_verification.utils.files import FileFormatError, read_jsonl
from rag_claim_verification.utils.hashing import combine_hashes, hash_mapping, sha256_file


def parse_manifest(path: Path) -> list[Document]:
    """Parse a non-empty JSONL manifest and reject duplicate document IDs."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ManifestError(f"Manifest does not exist: {resolved}")
    try:
        raw_records = read_jsonl(resolved)
    except FileFormatError as exc:
        raise ManifestError(str(exc)) from exc
    if not raw_records:
        raise ManifestError(f"Manifest contains no documents: {resolved}")

    documents: list[Document] = []
    seen: dict[str, int] = {}
    for line_number, raw in raw_records:
        try:
            document = Document.model_validate(raw)
        except ValidationError as exc:
            raise ManifestError(f"Invalid document at {resolved}:{line_number}:\n{exc}") from exc
        if document.document_id in seen:
            first_line = seen[document.document_id]
            raise ManifestError(
                f"Duplicate document_id {document.document_id!r} at {resolved}:{line_number} "
                f"(first declared at line {first_line})"
            )
        seen[document.document_id] = line_number
        documents.append(document)
    return documents


def resolve_document_path(document: Document, manifest_path: Path) -> Path:
    """Resolve a document path relative to the manifest that declares it."""

    if document.file_path.is_absolute():
        return document.file_path.resolve()
    return (manifest_path.resolve().parent / document.file_path).resolve()


def compute_corpus_hash(manifest_path: Path) -> str:
    """Hash manifest metadata and every referenced document's content."""

    documents = parse_manifest(manifest_path)
    component_hashes = [sha256_file(manifest_path.resolve())]
    for document in sorted(documents, key=lambda item: item.document_id):
        file_path = resolve_document_path(document, manifest_path)
        if not file_path.is_file():
            raise ManifestError(f"Document file does not exist: {file_path}")
        component_hashes.append(
            hash_mapping(
                {
                    "document": document.model_dump(mode="json"),
                    "content_hash": sha256_file(file_path),
                }
            )
        )
    return combine_hashes(*component_hashes)


def validate_noisy_superset(clean_manifest: Path, noisy_manifest: Path) -> None:
    """Ensure the noisy corpus contains byte-identical content for every clean document ID."""

    clean = {document.document_id: document for document in parse_manifest(clean_manifest)}
    noisy = {document.document_id: document for document in parse_manifest(noisy_manifest)}
    missing = sorted(set(clean) - set(noisy))
    if missing:
        raise ManifestError("Noisy manifest is missing clean document IDs: " + ", ".join(missing))
    mismatched: list[str] = []
    metadata_mismatched: list[str] = []
    for document_id, clean_document in clean.items():
        clean_path = resolve_document_path(clean_document, clean_manifest)
        noisy_path = resolve_document_path(noisy[document_id], noisy_manifest)
        if not clean_path.is_file() or not noisy_path.is_file():
            missing_path = clean_path if not clean_path.is_file() else noisy_path
            raise ManifestError(f"Document file does not exist: {missing_path}")
        if sha256_file(clean_path) != sha256_file(noisy_path):
            mismatched.append(document_id)
        clean_metadata = clean_document.model_dump(
            mode="json", exclude={"file_path", "corpus_tags"}
        )
        noisy_metadata = noisy[document_id].model_dump(
            mode="json", exclude={"file_path", "corpus_tags"}
        )
        if clean_metadata != noisy_metadata:
            metadata_mismatched.append(document_id)
    if mismatched:
        raise ManifestError(
            "Noisy corpus changed clean document content for IDs: " + ", ".join(mismatched)
        )
    if metadata_mismatched:
        raise ManifestError(
            "Noisy corpus changed clean document metadata for IDs: "
            + ", ".join(metadata_mismatched)
        )
