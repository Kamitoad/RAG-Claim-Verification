"""Command-line workflows for corpus preparation, verification, and evaluation."""

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from rag_claim_verification.config import (
    LIGHTRAG_VERSION,
    BenchmarkConfig,
    CorpusConfig,
    corpus_index_config_hash,
    load_benchmark_config,
    load_corpus_config,
)
from rag_claim_verification.evaluation.benchmark import (
    BenchmarkRunner,
    DefaultComponentFactory,
    load_claims,
)
from rag_claim_verification.evaluation.evaluate import evaluate_run
from rag_claim_verification.ingestion.loader import load_documents
from rag_claim_verification.ingestion.manifest import (
    compute_corpus_hash,
    validate_noisy_superset,
)
from rag_claim_verification.ingestion.service import IngestionService
from rag_claim_verification.logging_config import configure_logging
from rag_claim_verification.models.claim import Claim
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.files import read_yaml
from rag_claim_verification.utils.hashing import sha256_text
from rag_claim_verification.verification.prompt_builder import PromptBuilder
from rag_claim_verification.verification.verifier import ClaimVerifier

app = typer.Typer(
    name="rag-claim-verification",
    help="Evidence-grounded claim verification and controlled RAG benchmarks.",
    no_args_is_help=True,
)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Configure shared CLI behavior."""

    configure_logging(verbose)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _fail(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2)


@app.command("validate-config")
def validate_config_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Validate a benchmark or corpus YAML configuration."""

    try:
        raw = read_yaml(config.resolve())
        resolved: BenchmarkConfig | CorpusConfig
        if "conditions" in raw:
            resolved = load_benchmark_config(config)
            PromptBuilder(resolved.prompts)
            load_claims(resolved.claims_file)
            comparable_retriever: tuple[str, dict[str, Any]] | None = None
            for condition in resolved.conditions:
                if condition.corpus_config is None:
                    continue
                corpus = load_corpus_config(condition.corpus_config)
                load_documents(corpus.manifest_path)
                if corpus.clean_manifest_path is not None:
                    validate_noisy_superset(corpus.clean_manifest_path, corpus.manifest_path)
                if resolved.enforce_comparability:
                    settings = corpus.retriever.model_dump(mode="json")
                    settings.pop("working_directory", None)
                    if comparable_retriever is None:
                        comparable_retriever = (condition.id, settings)
                    elif settings != comparable_retriever[1]:
                        raise ValueError(
                            f"RAG conditions {comparable_retriever[0]!r} and "
                            f"{condition.id!r} use different retriever/index settings"
                        )
        else:
            resolved = load_corpus_config(config)
            load_documents(resolved.manifest_path)
            if resolved.clean_manifest_path is not None:
                validate_noisy_superset(resolved.clean_manifest_path, resolved.manifest_path)
            if resolved.prompts is not None:
                PromptBuilder(resolved.prompts)
        _print_json(
            {
                "status": "valid",
                "config_type": type(resolved).__name__,
                "config": resolved.model_dump(mode="json"),
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("validate-corpus")
def validate_corpus_command(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    clean_manifest: Annotated[
        Path | None,
        typer.Option(
            "--clean-manifest",
            exists=True,
            dir_okay=False,
            help="Optionally verify that this manifest is a superset of a clean corpus.",
        ),
    ] = None,
) -> None:
    """Validate manifest records and every referenced UTF-8 document."""

    try:
        documents = load_documents(manifest)
        if clean_manifest is not None:
            validate_noisy_superset(clean_manifest, manifest)
        _print_json(
            {
                "status": "valid",
                "manifest": str(manifest.resolve()),
                "document_count": len(documents),
                "document_ids": [item.metadata.document_id for item in documents],
                "corpus_hash": compute_corpus_hash(manifest),
                "clean_superset_check": clean_manifest is not None,
            }
        )
    except Exception as exc:
        _fail(exc)


@app.command("ingest")
def ingest_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Validate and ingest one configured corpus into its LightRAG working directory."""

    try:
        corpus = load_corpus_config(config)
        if corpus.retriever.type != "lightrag":
            raise ValueError("ingest requires a corpus with retriever.type=lightrag")
        if corpus.clean_manifest_path is not None:
            validate_noisy_superset(corpus.clean_manifest_path, corpus.manifest_path)
        documents = load_documents(corpus.manifest_path)
        adapter = LightRAGAdapter(corpus.retriever, documents)
        working_directory = corpus.retriever.working_directory
        if working_directory is None:
            raise ValueError("LightRAG working_directory is required")
        service = IngestionService(
            corpus_id=corpus.corpus_id,
            manifest_path=corpus.manifest_path,
            manifest_hash=compute_corpus_hash(corpus.manifest_path),
            config_hash=corpus_index_config_hash(corpus),
            working_directory=working_directory,
            lightrag_version=LIGHTRAG_VERSION,
            ingestor=adapter,
        )

        async def run_ingestion() -> dict[str, Any]:
            try:
                summary = await service.run()
                return summary.model_dump(mode="json")
            finally:
                await adapter.close()

        _print_json(asyncio.run(run_ingestion()))
    except Exception as exc:
        _fail(exc)


@app.command("verify")
def verify_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    claim_text: Annotated[str, typer.Option("--claim", min=1)],
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 5,
) -> None:
    """Retrieve evidence and verify one ad-hoc claim."""

    try:
        corpus = load_corpus_config(config)
        if corpus.verification_model is None or corpus.prompts is None:
            raise ValueError("verify requires verification_model and prompts in the corpus config")
        factory = DefaultComponentFactory()
        retriever = factory.create_retriever(corpus)
        client = factory.create_llm(corpus.verification_model)
        verifier = ClaimVerifier(client, PromptBuilder(corpus.prompts))
        claim = Claim(
            claim_id=f"adhoc_{sha256_text(claim_text)[:12]}",
            claim=claim_text,
        )

        async def run_verification() -> dict[str, Any]:
            try:
                started = time.perf_counter()
                evidence = await retriever.retrieve(claim.claim, top_k)
                retrieval_latency_ms = round((time.perf_counter() - started) * 1000)
                prediction = await verifier.verify(
                    claim,
                    evidence,
                    condition=corpus.corpus_id,
                    retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_supports_document_ids=(
                        retriever.supports_document_ids
                        and all(item.document_id is not None for item in evidence)
                    ),
                )
                return prediction.model_copy(
                    update={"latency_ms": round((time.perf_counter() - started) * 1000)}
                ).model_dump(mode="json")
            finally:
                await retriever.close()
                await client.close()

        payload = asyncio.run(run_verification())
        _print_json(payload)
        if payload.get("predicted_label") is None:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


@app.command("benchmark")
def benchmark_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    claims: Annotated[
        Path | None,
        typer.Option(
            "--claims",
            exists=True,
            dir_okay=False,
            help="Override claims_file from the benchmark configuration.",
        ),
    ] = None,
) -> None:
    """Execute every configured condition over the same ordered claim set."""

    try:
        benchmark_config = load_benchmark_config(config)
        if claims is not None:
            benchmark_config = benchmark_config.model_copy(update={"claims_file": claims.resolve()})
        run_directory = asyncio.run(
            BenchmarkRunner(config=benchmark_config, config_path=config).run()
        )
        metadata = read_yaml(run_directory / "metadata.json")
        _print_json(
            {
                "status": metadata.get("status", "completed"),
                "run_directory": str(run_directory),
            }
        )
        if metadata.get("failed_prediction_count", 0) != 0:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc)


@app.command("evaluate")
def evaluate_command(
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False)],
) -> None:
    """Regenerate metrics and reports from an existing predictions file."""

    try:
        metrics = evaluate_run(run_dir)
        _print_json({"status": "evaluated", "metrics": metrics})
    except Exception as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
