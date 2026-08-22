"""Safe derivation of one LightRAG index from an immutable validated base index."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rag_claim_verification.config import (
    LIGHTRAG_VERSION,
    CorpusConfig,
    corpus_index_config_hash,
    load_corpus_config,
    retriever_index_config_hash,
)
from rag_claim_verification.errors import ManifestError
from rag_claim_verification.ingestion.manifest import (
    compute_corpus_hash,
    validate_noisy_superset,
)
from rag_claim_verification.ingestion.service import (
    DERIVED_BASE_METADATA_NAME,
    INGESTION_METADATA_NAME,
    DerivedFromIndex,
    derived_from_summary,
    read_ingestion_summary,
    validate_ingested_index,
)


@dataclass(frozen=True, slots=True)
class ResolvedDerivation:
    """Validated base corpus configuration and its exact persisted provenance."""

    base_config: CorpusConfig
    provenance: DerivedFromIndex


def resolve_derivation(config: CorpusConfig) -> ResolvedDerivation | None:
    """Validate a declared base config/index and construct its exact provenance record."""

    if config.derived_from is None:
        return None
    base = load_corpus_config(config.derived_from.corpus_config)
    if base.derived_from is not None:
        raise ManifestError("Nested derived_from index chains are not supported")
    if base.retriever.type != "lightrag":
        raise ManifestError("derived_from base must use retriever.type=lightrag")
    if config.clean_manifest_path is None:
        raise ManifestError("derived_from corpus requires clean_manifest_path")
    if compute_corpus_hash(config.clean_manifest_path) != compute_corpus_hash(base.manifest_path):
        raise ManifestError(
            "Derived corpus clean_manifest_path does not identify the declared base corpus"
        )
    validate_noisy_superset(base.manifest_path, config.manifest_path)
    if retriever_index_config_hash(config) != retriever_index_config_hash(base):
        raise ManifestError(
            "Derived corpus must use the same index-producing retriever settings as its base"
        )

    base_working = base.retriever.working_directory
    if base_working is None:
        raise ManifestError("derived_from base has no LightRAG working_directory")
    validate_ingested_index(
        corpus_id=base.corpus_id,
        manifest_hash=compute_corpus_hash(base.manifest_path),
        config_hash=corpus_index_config_hash(base),
        working_directory=base_working,
        lightrag_version=LIGHTRAG_VERSION,
    )
    metadata_path = base_working / INGESTION_METADATA_NAME
    summary = read_ingestion_summary(metadata_path)
    return ResolvedDerivation(
        base_config=base,
        provenance=derived_from_summary(summary, metadata_path=metadata_path),
    )


def prepare_derived_working_directory(
    *,
    base_working_directory: Path,
    target_working_directory: Path,
) -> None:
    """Atomically copy a validated base index into a never-before-used target directory."""

    base = base_working_directory.resolve()
    target = target_working_directory.resolve()
    if base == target:
        raise ManifestError("Derived and base indices must use different working directories")
    if target.exists():
        raise ManifestError(f"Refusing to overwrite existing derived index directory: {target}")
    source_metadata = base / INGESTION_METADATA_NAME
    if not source_metadata.is_file():
        raise ManifestError(f"Base index has no ingestion metadata: {source_metadata}")

    temporary = target.with_name(f".{target.name}.deriving")
    if temporary.exists():
        raise ManifestError(
            f"Refusing to overwrite existing derivation staging directory: {temporary}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(base, temporary, copy_function=shutil.copy2)
        copied_metadata = temporary / INGESTION_METADATA_NAME
        copied_metadata.replace(temporary / DERIVED_BASE_METADATA_NAME)
        temporary.rename(target)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
