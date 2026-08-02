"""Complete controlled benchmark workflow and run provenance."""

import importlib.metadata
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from rag_claim_verification.config import (
    LIGHTRAG_VERSION,
    BenchmarkCondition,
    BenchmarkConfig,
    CorpusConfig,
    OpenAICompatibleConfig,
    corpus_index_config_hash,
    load_corpus_config,
)
from rag_claim_verification.errors import ConfigurationError
from rag_claim_verification.evaluation.reporting import write_evaluation_artifacts
from rag_claim_verification.ingestion.loader import load_documents
from rag_claim_verification.ingestion.manifest import (
    compute_corpus_hash,
    validate_noisy_superset,
)
from rag_claim_verification.ingestion.service import validate_ingested_index
from rag_claim_verification.llm.base import LLMClient
from rag_claim_verification.llm.openai_compatible import OpenAICompatibleClient
from rag_claim_verification.models.claim import Claim
from rag_claim_verification.models.prediction import Prediction
from rag_claim_verification.models.run import (
    ConditionMetadata,
    RunMetadata,
    RunStatus,
)
from rag_claim_verification.retrieval.base import Retriever
from rag_claim_verification.retrieval.in_memory_retriever import InMemoryKeywordRetriever
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.files import (
    FileFormatError,
    ensure_new_directory,
    read_jsonl,
    write_json,
    write_jsonl,
    write_yaml,
)
from rag_claim_verification.utils.hashing import sha256_file
from rag_claim_verification.verification.baseline import BaselineVerifier
from rag_claim_verification.verification.prompt_builder import PromptBuilder
from rag_claim_verification.verification.verifier import ClaimVerifier


class ComponentFactory(Protocol):
    """Dependency-injection seam for external and fully offline benchmark components."""

    def create_llm(self, config: OpenAICompatibleConfig) -> LLMClient:
        """Create a verification model client."""

    def create_retriever(self, config: CorpusConfig) -> Retriever:
        """Create a retriever for one resolved corpus configuration."""


class DefaultComponentFactory:
    """Build production adapters selected by validated configuration."""

    def create_llm(self, config: OpenAICompatibleConfig) -> LLMClient:
        """Create the OpenAI-compatible verification client."""

        return OpenAICompatibleClient(config)

    def create_retriever(self, config: CorpusConfig) -> Retriever:
        """Build either the pinned LightRAG adapter or deterministic keyword retriever."""

        documents = load_documents(config.manifest_path)
        if config.retriever.type == "in_memory":
            return InMemoryKeywordRetriever(documents)
        working = config.retriever.working_directory
        if working is None:
            raise ConfigurationError("LightRAG working_directory is required")
        validate_ingested_index(
            corpus_id=config.corpus_id,
            manifest_hash=compute_corpus_hash(config.manifest_path),
            config_hash=corpus_index_config_hash(config),
            working_directory=working,
            lightrag_version=LIGHTRAG_VERSION,
        )
        return LightRAGAdapter(config.retriever, documents)


def load_claims(path: Path, *, require_gold: bool = True) -> list[Claim]:
    """Load a non-empty JSONL claim set and reject duplicate claim IDs."""

    if not path.is_file():
        raise ConfigurationError(f"Claims file does not exist: {path}")
    try:
        records = read_jsonl(path)
    except FileFormatError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not records:
        raise ConfigurationError(f"Claims file contains no claims: {path}")
    claims: list[Claim] = []
    seen: dict[str, int] = {}
    for line_number, raw in records:
        try:
            claim = Claim.model_validate(raw)
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid claim at {path}:{line_number}:\n{exc}") from exc
        if claim.claim_id in seen:
            raise ConfigurationError(
                f"Duplicate claim_id {claim.claim_id!r} at {path}:{line_number} "
                f"(first declared at line {seen[claim.claim_id]})"
            )
        if require_gold and claim.gold_label is None:
            raise ConfigurationError(
                f"Benchmark claim {claim.claim_id!r} has no gold_label at {path}:{line_number}"
            )
        seen[claim.claim_id] = line_number
        claims.append(claim)
    return claims


class BenchmarkRunner:
    """Run each condition over the same ordered claims and persist complete artifacts."""

    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        config_path: Path,
        component_factory: ComponentFactory | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._config_path = config_path.resolve()
        self._factory = component_factory or DefaultComponentFactory()
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run(self) -> Path:
        """Execute all conditions, isolating per-prediction failures where possible."""

        claims = load_claims(self._config.claims_file)
        prompt_builder = PromptBuilder(self._config.prompts)
        corpora = self._load_and_validate_corpora()
        run_id = self._make_run_id()
        run_directory = self._config.output_directory / run_id
        ensure_new_directory(run_directory)
        metadata = self._build_metadata(run_id, prompt_builder, corpora)
        write_json(run_directory / "metadata.json", metadata.model_dump(mode="json"))
        write_yaml(run_directory / "resolved_config.yaml", self._resolved_config(corpora))

        predictions: list[Prediction] = []
        try:
            for condition in self._config.conditions:
                condition_predictions = await self._run_condition(
                    condition, claims, prompt_builder, corpora
                )
                predictions.extend(condition_predictions)
            write_jsonl(
                run_directory / "predictions.jsonl",
                (prediction.model_dump(mode="json") for prediction in predictions),
            )
            write_evaluation_artifacts(run_directory, predictions)
            failed_count = sum(item.predicted_label is None for item in predictions)
            metadata = metadata.model_copy(
                update={
                    "status": (
                        RunStatus.COMPLETED_WITH_ERRORS if failed_count else RunStatus.COMPLETED
                    ),
                    "ended_at": self._clock(),
                    "prediction_count": len(predictions),
                    "successful_prediction_count": len(predictions) - failed_count,
                    "failed_prediction_count": failed_count,
                }
            )
            write_json(run_directory / "metadata.json", metadata.model_dump(mode="json"))
            return run_directory
        except Exception:
            metadata = metadata.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "ended_at": self._clock(),
                    "prediction_count": len(predictions),
                    "successful_prediction_count": sum(
                        item.predicted_label is not None for item in predictions
                    ),
                    "failed_prediction_count": sum(
                        item.predicted_label is None for item in predictions
                    ),
                }
            )
            write_json(run_directory / "metadata.json", metadata.model_dump(mode="json"))
            raise

    def _load_and_validate_corpora(self) -> dict[str, CorpusConfig]:
        corpora: dict[str, CorpusConfig] = {}
        working_directories: dict[Path, str] = {}
        comparable_retriever: tuple[str, dict[str, object]] | None = None
        for condition in self._config.conditions:
            if condition.corpus_config is None:
                continue
            corpus = load_corpus_config(condition.corpus_config)
            load_documents(corpus.manifest_path)
            if corpus.clean_manifest_path is not None:
                validate_noisy_superset(corpus.clean_manifest_path, corpus.manifest_path)
            if self._config.enforce_comparability:
                retriever_settings = corpus.retriever.model_dump(mode="json")
                retriever_settings.pop("working_directory", None)
                if comparable_retriever is None:
                    comparable_retriever = (condition.id, retriever_settings)
                elif retriever_settings != comparable_retriever[1]:
                    raise ConfigurationError(
                        f"RAG conditions {comparable_retriever[0]!r} and {condition.id!r} "
                        "use different retriever/index settings beyond working_directory"
                    )
            if corpus.retriever.type == "lightrag":
                working = corpus.retriever.working_directory
                if working is None:
                    raise ConfigurationError("LightRAG working_directory is required")
                validate_ingested_index(
                    corpus_id=corpus.corpus_id,
                    manifest_hash=compute_corpus_hash(corpus.manifest_path),
                    config_hash=corpus_index_config_hash(corpus),
                    working_directory=working,
                    lightrag_version=LIGHTRAG_VERSION,
                )
            working = corpus.retriever.working_directory
            if working is not None:
                prior = working_directories.get(working)
                if prior is not None and prior != corpus.corpus_id:
                    raise ConfigurationError(
                        f"Corpora {prior!r} and {corpus.corpus_id!r} share working "
                        f"directory {working}"
                    )
                working_directories[working] = corpus.corpus_id
            corpora[condition.id] = corpus
        return corpora

    async def _run_condition(
        self,
        condition: BenchmarkCondition,
        claims: list[Claim],
        prompt_builder: PromptBuilder,
        corpora: dict[str, CorpusConfig],
    ) -> list[Prediction]:
        model_config = self._config.model_configs[condition.model_config_id]
        try:
            client = self._factory.create_llm(model_config)
        except Exception as exc:
            return [self._condition_error(claim, condition, exc) for claim in claims]
        verifier = ClaimVerifier(client, prompt_builder)
        retriever: Retriever | None = None
        try:
            if condition.mode == "baseline":
                baseline = BaselineVerifier(verifier)
                return [await baseline.verify(claim, condition=condition.id) for claim in claims]
            corpus = corpora[condition.id]
            try:
                retriever = self._factory.create_retriever(corpus)
            except Exception as exc:
                return [self._condition_error(claim, condition, exc) for claim in claims]
            predictions: list[Prediction] = []
            for claim in claims:
                started = time.perf_counter()
                try:
                    evidence = await retriever.retrieve(claim.claim, condition.top_k or 1)
                except Exception as exc:
                    retrieval_ms = round((time.perf_counter() - started) * 1000)
                    predictions.append(
                        self._condition_error(
                            claim, condition, exc, retrieval_latency_ms=retrieval_ms
                        )
                    )
                    continue
                retrieval_ms = round((time.perf_counter() - started) * 1000)
                supports_ids = retriever.supports_document_ids and all(
                    item.document_id is not None for item in evidence
                )
                predictions.append(
                    await verifier.verify(
                        claim,
                        evidence,
                        condition=condition.id,
                        retrieval_latency_ms=retrieval_ms,
                        retrieval_supports_document_ids=supports_ids,
                    )
                )
            return predictions
        finally:
            if retriever is not None:
                await retriever.close()
            await client.close()

    @staticmethod
    def _condition_error(
        claim: Claim,
        condition: BenchmarkCondition,
        exc: Exception,
        *,
        retrieval_latency_ms: int = 0,
    ) -> Prediction:
        return Prediction(
            claim_id=claim.claim_id,
            condition=condition.id,
            predicted_label=None,
            reason=None,
            evidence=[],
            cited_document_ids=[],
            latency_ms=retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            raw_model_output=None,
            parse_error=None,
            error=f"{type(exc).__name__}: {exc}",
            gold_label=claim.gold_label,
            gold_document_ids=claim.gold_document_ids,
            retrieval_supports_document_ids=False,
            verification_mode="baseline" if condition.mode == "baseline" else "rag",
        )

    def _build_metadata(
        self,
        run_id: str,
        prompts: PromptBuilder,
        corpora: dict[str, CorpusConfig],
    ) -> RunMetadata:
        condition_metadata: list[ConditionMetadata] = []
        for condition in self._config.conditions:
            model = self._config.model_configs[condition.model_config_id]
            corpus = corpora.get(condition.id)
            condition_metadata.append(
                ConditionMetadata(
                    condition_id=condition.id,
                    mode=condition.mode,
                    model_name=model.model,
                    model_endpoint=model.sanitized_endpoint(),
                    temperature=model.temperature,
                    retriever_type=(
                        (
                            "in_memory_keyword"
                            if corpus.retriever.type == "in_memory"
                            else f"lightrag_sdk_{LIGHTRAG_VERSION}"
                        )
                        if corpus is not None
                        else "none"
                    ),
                    top_k=condition.top_k,
                    corpus_id=corpus.corpus_id if corpus is not None else None,
                    manifest_hash=(
                        compute_corpus_hash(corpus.manifest_path) if corpus is not None else None
                    ),
                )
            )
        return RunMetadata(
            run_id=run_id,
            experiment_name=self._config.experiment_name,
            status=RunStatus.RUNNING,
            started_at=self._clock(),
            git_commit=_git_commit(self._config_path.parent),
            python_version=sys.version,
            platform=platform.platform(),
            package_versions=_package_versions(),
            config_file=str(self._config_path),
            config_hash=sha256_file(self._config_path),
            claims_file=str(self._config.claims_file),
            claims_hash=sha256_file(self._config.claims_file),
            prompt_hash=prompts.prompt_hash,
            prompt_version=prompts.version,
            conditions=condition_metadata,
        )

    def _resolved_config(self, corpora: dict[str, CorpusConfig]) -> dict[str, object]:
        return {
            "benchmark": self._config.model_dump(mode="json"),
            "corpora": {
                condition_id: corpus.model_dump(mode="json")
                for condition_id, corpus in corpora.items()
            },
        }

    def _make_run_id(self) -> str:
        timestamp = self._clock().astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _git_commit(start_directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _package_versions() -> dict[str, str | None]:
    names = [
        "rag-claim-verification",
        "pydantic",
        "PyYAML",
        "typer",
        "httpx",
        "lightrag-hku",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
