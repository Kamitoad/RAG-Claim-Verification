"""Compare LightRAG query modes on one validated F1 pilot index without verification."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from pydantic import Field

from rag_claim_verification.config import (
    LIGHTRAG_VERSION,
    corpus_index_config_hash,
    load_corpus_config,
)
from rag_claim_verification.evaluation.benchmark import load_claims
from rag_claim_verification.evaluation.retrieval_diagnostic import (
    QueryMode,
    RetrievalDiagnosticCase,
    RetrievalModeSummary,
    build_diagnostic_case,
    summarize_mode,
)
from rag_claim_verification.ingestion.derivation import resolve_derivation
from rag_claim_verification.ingestion.loader import load_documents
from rag_claim_verification.ingestion.manifest import (
    compute_corpus_hash,
    validate_noisy_superset,
)
from rag_claim_verification.ingestion.service import (
    INGESTION_METADATA_NAME,
    validate_ingested_index,
)
from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.models.claim import Claim
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.files import ensure_new_directory, write_json
from rag_claim_verification.utils.hashing import sha256_file

DEFAULT_MODES: tuple[QueryMode, ...] = ("hybrid", "naive", "mix")


class RetrievalDiagnosticRun(StrictModel):
    """Persisted, re-inspectable result of one retrieval-only diagnostic."""

    schema_version: Literal[1] = 1
    run_id: str
    generated_at: datetime
    corpus_id: str
    original_index_query_mode: QueryMode
    diagnostic_query_modes: list[QueryMode] = Field(min_length=1)
    top_k: int = Field(gt=0)
    corpus_config: str
    claims_file: str
    input_hashes: dict[str, str]
    preceding_corpus_id: str | None = None
    preceding_config: str | None = None
    preceding_cases: list[RetrievalDiagnosticCase] = Field(default_factory=list)
    clean_document_ids: list[str]
    noise_document_ids: list[str]
    cases: list[RetrievalDiagnosticCase]
    mode_summaries: list[RetrievalModeSummary]
    failure_count: int = Field(ge=0)


def _run_id(now: datetime) -> str:
    timestamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"retrieval-{timestamp}-{uuid.uuid4().hex[:8]}"


async def _diagnose_mode(
    *,
    mode: QueryMode,
    claims: list[Claim],
    adapter: LightRAGAdapter,
    noise_ids: set[str],
    top_k: int,
) -> list[RetrievalDiagnosticCase]:
    cases: list[RetrievalDiagnosticCase] = []
    try:
        for claim in claims:
            started = perf_counter()
            try:
                evidence = await adapter.retrieve(claim.claim, top_k=top_k)
                error: Exception | None = None
            except Exception as exc:  # external SDK boundary; persist the actual failure
                evidence = []
                error = exc
            latency_ms = round((perf_counter() - started) * 1000)
            cases.append(
                build_diagnostic_case(
                    mode=mode,
                    claim=claim,
                    evidence=evidence,
                    noise_ids=noise_ids,
                    retrieval_latency_ms=latency_ms,
                    error=error,
                )
            )
    finally:
        await adapter.close()
    return cases


async def diagnose(
    *,
    project_root: Path,
    config_path: Path,
    claims_path: Path,
    output_root: Path,
    modes: list[QueryMode],
    top_k: int,
    preceding_config_path: Path | None = None,
) -> Path:
    """Validate one existing index, run query-mode overrides, and persist observations."""

    if len(set(modes)) != len(modes):
        raise ValueError("diagnostic query modes must be unique")
    config = load_corpus_config(config_path)
    if config.retriever.type != "lightrag":
        raise ValueError("retrieval diagnostic requires retriever.type=lightrag")
    if config.clean_manifest_path is None:
        raise ValueError("retrieval diagnostic requires clean_manifest_path")
    working = config.retriever.working_directory
    if working is None:
        raise ValueError("retrieval diagnostic requires a LightRAG working_directory")

    validate_noisy_superset(config.clean_manifest_path, config.manifest_path)
    derivation = resolve_derivation(config)
    validate_ingested_index(
        corpus_id=config.corpus_id,
        manifest_hash=compute_corpus_hash(config.manifest_path),
        config_hash=corpus_index_config_hash(config),
        working_directory=working,
        lightrag_version=LIGHTRAG_VERSION,
        derived_from=(derivation.provenance if derivation is not None else None),
    )

    documents = load_documents(config.manifest_path)
    clean_documents = load_documents(config.clean_manifest_path)
    clean_ids = {document.metadata.document_id for document in clean_documents}
    all_ids = {document.metadata.document_id for document in documents}
    noise_ids = all_ids - clean_ids
    claims = load_claims(claims_path)

    input_hashes = {
        "corpus_config": sha256_file(config_path),
        "manifest": sha256_file(config.manifest_path),
        "corpus_content": compute_corpus_hash(config.manifest_path),
        "claims": sha256_file(claims_path),
        "ingestion_metadata": sha256_file(working / INGESTION_METADATA_NAME),
        "diagnostic_script": sha256_file(Path(__file__)),
    }
    preceding_corpus_id: str | None = None
    preceding_config: str | None = None
    preceding_cases: list[RetrievalDiagnosticCase] = []
    if preceding_config_path is not None:
        prior = load_corpus_config(preceding_config_path)
        if prior.retriever.type != "lightrag":
            raise ValueError("preceding diagnostic config requires retriever.type=lightrag")
        prior_working = prior.retriever.working_directory
        if prior_working is None:
            raise ValueError("preceding diagnostic config requires a working_directory")
        prior_derivation = resolve_derivation(prior)
        validate_ingested_index(
            corpus_id=prior.corpus_id,
            manifest_hash=compute_corpus_hash(prior.manifest_path),
            config_hash=corpus_index_config_hash(prior),
            working_directory=prior_working,
            lightrag_version=LIGHTRAG_VERSION,
            derived_from=(prior_derivation.provenance if prior_derivation is not None else None),
        )
        prior_documents = load_documents(prior.manifest_path)
        preceding_cases = await _diagnose_mode(
            mode=prior.retriever.query_mode,
            claims=claims,
            adapter=LightRAGAdapter(prior.retriever, prior_documents),
            noise_ids=set(),
            top_k=top_k,
        )
        preceding_corpus_id = prior.corpus_id
        preceding_config = str(preceding_config_path.resolve().relative_to(project_root.resolve()))
        input_hashes.update(
            {
                "preceding_corpus_config": sha256_file(preceding_config_path),
                "preceding_manifest": sha256_file(prior.manifest_path),
                "preceding_corpus_content": compute_corpus_hash(prior.manifest_path),
                "preceding_ingestion_metadata": sha256_file(
                    prior_working / INGESTION_METADATA_NAME
                ),
            }
        )

    cases: list[RetrievalDiagnosticCase] = []
    for mode in modes:
        runtime_retriever = config.retriever.model_copy(update={"query_mode": mode})
        adapter = LightRAGAdapter(runtime_retriever, documents)
        cases.extend(
            await _diagnose_mode(
                mode=mode,
                claims=claims,
                adapter=adapter,
                noise_ids=noise_ids,
                top_k=top_k,
            )
        )

    now = datetime.now(UTC)
    run_id = _run_id(now)
    run_directory = output_root / run_id
    ensure_new_directory(run_directory)
    result = RetrievalDiagnosticRun(
        run_id=run_id,
        generated_at=now,
        corpus_id=config.corpus_id,
        original_index_query_mode=config.retriever.query_mode,
        diagnostic_query_modes=modes,
        top_k=top_k,
        corpus_config=str(config_path.resolve().relative_to(project_root.resolve())),
        claims_file=str(claims_path.resolve().relative_to(project_root.resolve())),
        input_hashes=input_hashes,
        preceding_corpus_id=preceding_corpus_id,
        preceding_config=preceding_config,
        preceding_cases=preceding_cases,
        clean_document_ids=sorted(clean_ids),
        noise_document_ids=sorted(noise_ids),
        cases=cases,
        mode_summaries=[summarize_mode(mode, cases) for mode in modes],
        failure_count=sum(case.error_type is not None for case in cases),
    )
    write_json(run_directory / "retrieval_diagnostic.json", result.model_dump(mode="json"))
    return run_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/f1_2023_pilot_noisy.yaml"),
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/ground_truth/f1_2023_pilot_gate.jsonl"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/retrieval_diagnostics"))
    parser.add_argument(
        "--preceding-config",
        type=Path,
        help="Optionally open and query another validated index first in the same process.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["local", "global", "hybrid", "naive", "mix"],
        dest="modes",
    )
    return parser.parse_args()


def main() -> int:
    """Run the local diagnostic and print the immutable result directory."""

    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    modes = [cast(QueryMode, mode) for mode in (args.modes or DEFAULT_MODES)]
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    run_directory = asyncio.run(
        diagnose(
            project_root=project_root,
            config_path=(project_root / args.config).resolve(),
            claims_path=(project_root / args.claims).resolve(),
            output_root=(project_root / args.output_root).resolve(),
            modes=modes,
            top_k=args.top_k,
            preceding_config_path=(
                (project_root / args.preceding_config).resolve()
                if args.preceding_config is not None
                else None
            ),
        )
    )
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
