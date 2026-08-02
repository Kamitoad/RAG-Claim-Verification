"""Validated corpus text loading."""

from dataclasses import dataclass
from pathlib import Path

from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.manifest import parse_manifest, resolve_document_path
from rag_claim_verification.models.document import Document
from rag_claim_verification.utils.files import FileFormatError, read_text
from rag_claim_verification.utils.hashing import sha256_text


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Validated document metadata coupled to its UTF-8 text."""

    metadata: Document
    text: str
    resolved_file_path: Path
    content_hash: str


def load_documents(manifest_path: Path) -> list[LoadedDocument]:
    """Load all manifest files, reporting every missing or empty file together."""

    documents = parse_manifest(manifest_path)
    problems: list[str] = []
    loaded: list[LoadedDocument] = []
    for document in documents:
        resolved_path = resolve_document_path(document, manifest_path)
        if not resolved_path.is_file():
            problems.append(f"{document.document_id}: missing file {resolved_path}")
            continue
        try:
            text = read_text(resolved_path)
        except FileFormatError as exc:
            problems.append(f"{document.document_id}: {exc}")
            continue
        if not text.strip():
            problems.append(f"{document.document_id}: empty document {resolved_path}")
            continue
        loaded.append(
            LoadedDocument(
                metadata=document,
                text=text,
                resolved_file_path=resolved_path,
                content_hash=sha256_text(text),
            )
        )
    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ManifestError(f"Corpus validation failed:\n{details}")
    return loaded
