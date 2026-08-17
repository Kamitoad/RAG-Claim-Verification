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
from rag_claim_verification.models.prediction import (
    CaseStatus,
    ParseStatus,
    Prediction,
    RetrievalStatus,
)
from rag_claim_verification.models.run import (
    CaseManifestRecord,
    ConditionMetadata,
    RunMetadata,
    RunStatus,
)
from rag_claim_verification.retrieval.base import Retriever
from rag_claim_verification.retrieval.in_memory_retriever import InMemoryKeywordRetriever
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.files import (
    FileFormatError,
    atomic_write_bytes,
    ensure_new_directory,
    read_jsonl,
    write_json,
    write_jsonl,
    write_yaml,
)
from rag_claim_verification.utils.hashing import hash_mapping, sha256_file
from rag_claim_verification.verification.baseline import BaselineVerifier
from rag_claim_verification.verification.prompt_builder import PromptBuilder
from rag_claim_verification.verification.verifier import ClaimVerifier

TIMING_DEFINITIONS = {
    "retrieval_latency_ms": (
        "Monotonic elapsed time spent awaiting Retriever.retrieve; zero when retrieval "
        "is not applicable or not started."
    ),
    "generation_latency_ms": (
        "Monotonic elapsed time spent awaiting the initial LLM generation call, including "
        "provider retries."
    ),
    "repair_latency_ms": (
        "Monotonic elapsed time spent awaiting the optional repair generation call, including "
        "provider retries."
    ),
    "latency_ms": (
        "End-to-end case time from retrieval start (RAG) or prompt rendering start (baseline) "
        "through final validated output or recorded failure."
    ),
    "benchmark_duration_ms": (
        "Monotonic elapsed time from immediately before the first condition until all raw and "
        "derived artifacts have been written."
    ),
}


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
        write_yaml(run_directory / "resolved_config.yaml", self._resolved_config(corpora))
        cases = self._build_case_manifest(claims)
        write_jsonl(
            run_directory / "case_manifest.jsonl",
            (case.model_dump(mode="json") for case in cases),
        )
        input_hashes = self._write_input_snapshots(run_directory, corpora)
        metadata = self._build_metadata(
            run_id,
            prompt_builder,
            corpora,
            run_directory=run_directory,
            cases=cases,
            input_hashes=input_hashes,
        )
        write_json(run_directory / "metadata.json", metadata.model_dump(mode="json"))

        predictions: list[Prediction] = []
        write_jsonl(run_directory / "predictions.jsonl", [])
        benchmark_started = time.perf_counter()

        def checkpoint(prediction: Prediction) -> None:
            predictions.append(prediction)
            write_jsonl(
                run_directory / "predictions.jsonl",
                (item.model_dump(mode="json") for item in predictions),
            )

        try:
            for condition in self._config.conditions:
                await self._run_condition(
                    condition,
                    claims,
                    prompt_builder,
                    corpora,
                    record_prediction=checkpoint,
                )
            write_evaluation_artifacts(run_directory, predictions, cases)
            failed_count = sum(item.predicted_label is None for item in predictions)
            duration_ms = round((time.perf_counter() - benchmark_started) * 1000)
            metadata = metadata.model_copy(
                update={
                    "status": (
                        RunStatus.COMPLETED_WITH_ERRORS if failed_count else RunStatus.COMPLETED
                    ),
                    "ended_at": self._clock(),
                    "prediction_count": len(predictions),
                    "predictions_hash": sha256_file(run_directory / "predictions.jsonl"),
                    "successful_prediction_count": len(predictions) - failed_count,
                    "failed_prediction_count": failed_count,
                    "missing_prediction_count": len(cases) - len(predictions),
                    "benchmark_duration_ms": duration_ms,
                }
            )
            write_json(run_directory / "metadata.json", metadata.model_dump(mode="json"))
            return run_directory
        except Exception as exc:
            recorded_ids = {prediction.case_id for prediction in predictions}
            for case in cases:
                if case.case_id not in recorded_ids:
                    predictions.append(self._run_error_prediction(case, exc))
            write_jsonl(
                run_directory / "predictions.jsonl",
                (item.model_dump(mode="json") for item in predictions),
            )
            duration_ms = round((time.perf_counter() - benchmark_started) * 1000)
            metadata = metadata.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "ended_at": self._clock(),
                    "prediction_count": len(predictions),
                    "predictions_hash": sha256_file(run_directory / "predictions.jsonl"),
                    "successful_prediction_count": sum(
                        item.predicted_label is not None for item in predictions
                    ),
                    "failed_prediction_count": sum(
                        item.predicted_label is None for item in predictions
                    ),
                    "missing_prediction_count": len(cases) - len(predictions),
                    "benchmark_duration_ms": duration_ms,
                    "run_error": f"{type(exc).__name__}: {exc}",
                    "run_error_stage": "benchmark",
                    "run_error_type": type(exc).__name__,
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
        *,
        record_prediction: Callable[[Prediction], None],
    ) -> None:
        model_config = self._config.model_configs[condition.model_config_id]
        try:
            client = self._factory.create_llm(model_config)
        except Exception as exc:
            for claim in claims:
                record_prediction(
                    self._condition_error(
                        claim,
                        condition,
                        exc,
                        case_status=CaseStatus.COMPONENT_ERROR,
                        error_stage="model_client_creation",
                        retrieval_status=(
                            RetrievalStatus.NOT_APPLICABLE
                            if condition.mode == "baseline"
                            else RetrievalStatus.NOT_STARTED
                        ),
                    )
                )
            return
        verifier = ClaimVerifier(client, prompt_builder)
        retriever: Retriever | None = None
        try:
            if condition.mode == "baseline":
                baseline = BaselineVerifier(verifier)
                for claim in claims:
                    record_prediction(await baseline.verify(claim, condition=condition.id))
                return
            corpus = corpora[condition.id]
            try:
                retriever = self._factory.create_retriever(corpus)
            except Exception as exc:
                for claim in claims:
                    record_prediction(
                        self._condition_error(
                            claim,
                            condition,
                            exc,
                            case_status=CaseStatus.COMPONENT_ERROR,
                            error_stage="retriever_creation",
                            retrieval_status=RetrievalStatus.NOT_STARTED,
                        )
                    )
                return
            for claim in claims:
                started = time.perf_counter()
                try:
                    evidence = await retriever.retrieve(claim.claim, condition.top_k or 1)
                except Exception as exc:
                    retrieval_ms = round((time.perf_counter() - started) * 1000)
                    record_prediction(
                        self._condition_error(
                            claim,
                            condition,
                            exc,
                            retrieval_latency_ms=retrieval_ms,
                            case_status=CaseStatus.RETRIEVAL_ERROR,
                            error_stage="retrieval",
                            retrieval_status=RetrievalStatus.ERROR,
                        )
                    )
                    continue
                retrieval_ms = round((time.perf_counter() - started) * 1000)
                supports_ids = retriever.supports_document_ids and all(
                    item.document_id is not None for item in evidence
                )
                prediction = await verifier.verify(
                    claim,
                    evidence,
                    condition=condition.id,
                    retrieval_latency_ms=retrieval_ms,
                    retrieval_supports_document_ids=supports_ids,
                    retrieval_status=(
                        RetrievalStatus.SUCCESS if evidence else RetrievalStatus.SUCCESS_EMPTY
                    ),
                )
                record_prediction(
                    prediction.model_copy(
                        update={"latency_ms": round((time.perf_counter() - started) * 1000)}
                    )
                )
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
        case_status: CaseStatus,
        error_stage: str,
        retrieval_status: RetrievalStatus,
    ) -> Prediction:
        return Prediction(
            case_id=f"{condition.id}:{claim.claim_id}",
            claim_id=claim.claim_id,
            claim=claim.claim,
            condition=condition.id,
            case_status=case_status,
            retrieval_status=retrieval_status,
            parse_status=ParseStatus.NOT_STARTED,
            predicted_label=None,
            reason=None,
            evidence=[],
            cited_document_ids=[],
            latency_ms=retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            raw_model_output=None,
            parse_error=None,
            error=f"{type(exc).__name__}: {exc}",
            error_stage=error_stage,
            error_type=type(exc).__name__,
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
        *,
        run_directory: Path,
        cases: list[CaseManifestRecord],
        input_hashes: dict[str, str],
    ) -> RunMetadata:
        git_root = _git_root(self._config_path.parent)
        lock_file = git_root / "uv.lock" if git_root is not None else None
        if lock_file is not None and not lock_file.is_file():
            lock_file = None
        condition_metadata: list[ConditionMetadata] = []
        for condition in self._config.conditions:
            model = self._config.model_configs[condition.model_config_id]
            corpus = corpora.get(condition.id)
            condition_metadata.append(
                ConditionMetadata(
                    condition_id=condition.id,
                    mode=condition.mode,
                    provider=model.provider,
                    model_name=model.model,
                    model_endpoint=model.sanitized_endpoint(),
                    temperature=model.temperature,
                    seed=model.seed,
                    timeout_seconds=model.timeout_seconds,
                    max_retries=model.max_retries,
                    request_json_object=model.request_json_object,
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
                    corpus_config_file=(
                        str(condition.corpus_config)
                        if condition.corpus_config is not None
                        else None
                    ),
                    corpus_config_hash=(
                        sha256_file(condition.corpus_config)
                        if condition.corpus_config is not None
                        else None
                    ),
                    manifest_file=(str(corpus.manifest_path) if corpus is not None else None),
                    manifest_file_hash=(
                        sha256_file(corpus.manifest_path) if corpus is not None else None
                    ),
                    corpus_hash=(
                        compute_corpus_hash(corpus.manifest_path) if corpus is not None else None
                    ),
                )
            )
        return RunMetadata(
            run_id=run_id,
            run_group_id=run_id,
            experiment_name=self._config.experiment_name,
            status=RunStatus.RUNNING,
            started_at=self._clock(),
            git_commit=_git_commit(self._config_path.parent),
            source_tree_hash=_source_tree_hash(git_root) if git_root is not None else None,
            python_version=sys.version,
            platform=platform.platform(),
            package_versions=_package_versions(),
            config_file=str(self._config_path),
            config_hash=sha256_file(self._config_path),
            resolved_config_hash=sha256_file(run_directory / "resolved_config.yaml"),
            claims_file=str(self._config.claims_file),
            claims_hash=sha256_file(self._config.claims_file),
            claims_snapshot_hash=input_hashes["claims_snapshot"],
            prompt_hash=prompts.prompt_hash,
            prompt_version=prompts.version,
            prompt_hashes=prompts.prompt_hashes,
            case_manifest_hash=sha256_file(run_directory / "case_manifest.jsonl"),
            git_dirty=_git_dirty(self._config_path.parent),
            dependency_lock_file=str(lock_file) if lock_file is not None else None,
            dependency_lock_hash=sha256_file(lock_file) if lock_file is not None else None,
            timing_definitions=TIMING_DEFINITIONS,
            conditions=condition_metadata,
            expected_prediction_count=len(cases),
            missing_prediction_count=len(cases),
        )

    def _build_case_manifest(self, claims: list[Claim]) -> list[CaseManifestRecord]:
        cases: list[CaseManifestRecord] = []
        for condition in self._config.conditions:
            for claim in claims:
                if claim.gold_label is None:
                    raise ConfigurationError(
                        f"Benchmark claim {claim.claim_id!r} has no gold label"
                    )
                cases.append(
                    CaseManifestRecord(
                        case_id=f"{condition.id}:{claim.claim_id}",
                        sequence=len(cases) + 1,
                        claim_id=claim.claim_id,
                        claim=claim.claim,
                        gold_label=claim.gold_label,
                        gold_document_ids=claim.gold_document_ids,
                        condition=condition.id,
                        verification_mode=condition.mode,
                    )
                )
        return cases

    @staticmethod
    def _run_error_prediction(case: CaseManifestRecord, exc: Exception) -> Prediction:
        return Prediction(
            case_id=case.case_id,
            claim_id=case.claim_id,
            claim=case.claim,
            condition=case.condition,
            case_status=CaseStatus.PIPELINE_ERROR,
            retrieval_status=(
                RetrievalStatus.NOT_APPLICABLE
                if case.verification_mode == "baseline"
                else RetrievalStatus.NOT_STARTED
            ),
            parse_status=ParseStatus.NOT_STARTED,
            predicted_label=None,
            reason=None,
            evidence=[],
            cited_document_ids=[],
            latency_ms=0,
            raw_model_output=None,
            parse_error=None,
            error=f"{type(exc).__name__}: {exc}",
            error_stage="benchmark",
            error_type=type(exc).__name__,
            gold_label=case.gold_label,
            gold_document_ids=case.gold_document_ids,
            retrieval_supports_document_ids=False,
            verification_mode=case.verification_mode,
        )

    def _write_input_snapshots(
        self, run_directory: Path, corpora: dict[str, CorpusConfig]
    ) -> dict[str, str]:
        inputs = run_directory / "inputs"
        prompts_directory = inputs / "prompts"
        corpus_configs_directory = inputs / "corpus_configs"
        manifests_directory = inputs / "manifests"
        ingestion_directory = inputs / "ingestion_metadata"
        for directory in (
            prompts_directory,
            corpus_configs_directory,
            manifests_directory,
            ingestion_directory,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        snapshots = {
            inputs / "benchmark.yaml": self._config_path,
            inputs / "claims.jsonl": self._config.claims_file,
            prompts_directory / "verification_system.txt": self._config.prompts.system_path,
            prompts_directory / "verification_user.txt": self._config.prompts.user_path,
            prompts_directory / "verification_repair.txt": self._config.prompts.repair_path,
        }
        for target, source in snapshots.items():
            atomic_write_bytes(target, source.read_bytes())

        hashes: dict[str, str] = {
            "benchmark_source": sha256_file(self._config_path),
            "claims_source": sha256_file(self._config.claims_file),
            "claims_snapshot": sha256_file(inputs / "claims.jsonl"),
        }
        for condition in self._config.conditions:
            corpus = corpora.get(condition.id)
            if corpus is None or condition.corpus_config is None:
                continue
            corpus_target = corpus_configs_directory / f"{condition.id}.yaml"
            manifest_target = manifests_directory / f"{condition.id}.jsonl"
            atomic_write_bytes(corpus_target, condition.corpus_config.read_bytes())
            atomic_write_bytes(manifest_target, corpus.manifest_path.read_bytes())
            hashes[f"corpus_config.{condition.id}"] = sha256_file(condition.corpus_config)
            hashes[f"manifest.{condition.id}"] = sha256_file(corpus.manifest_path)
            hashes[f"corpus_content.{condition.id}"] = compute_corpus_hash(corpus.manifest_path)
            for document in load_documents(corpus.manifest_path):
                hashes[f"document_content.{condition.id}.{document.metadata.document_id}"] = (
                    document.content_hash
                )
            if corpus.clean_manifest_path is not None:
                clean_target = manifests_directory / f"{condition.id}.clean_reference.jsonl"
                atomic_write_bytes(clean_target, corpus.clean_manifest_path.read_bytes())
                hashes[f"clean_manifest.{condition.id}"] = sha256_file(corpus.clean_manifest_path)
            working_directory = corpus.retriever.working_directory
            if working_directory is not None:
                ingestion_source = working_directory / "ragcv_ingestion_metadata.json"
                if ingestion_source.is_file():
                    ingestion_target = ingestion_directory / f"{condition.id}.json"
                    atomic_write_bytes(ingestion_target, ingestion_source.read_bytes())
                    hashes[f"ingestion_metadata.{condition.id}"] = sha256_file(ingestion_source)
        write_json(inputs / "hashes.json", hashes)
        return hashes

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


def _git_root(start_directory: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return Path(value).resolve() if value else None


def _source_tree_hash(repository_root: Path) -> str:
    sources = sorted((repository_root / "src").rglob("*.py"))
    pyproject = repository_root / "pyproject.toml"
    if pyproject.is_file():
        sources.append(pyproject)
    return hash_mapping(
        {source.relative_to(repository_root).as_posix(): sha256_file(source) for source in sources}
    )


def _git_dirty(start_directory: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=start_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _package_versions() -> dict[str, str | None]:
    names = [
        "rag-claim-verification",
        "pydantic",
        "PyYAML",
        "typer",
        "httpx",
        "lightrag-hku",
        "fastembed",
        "openai",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
