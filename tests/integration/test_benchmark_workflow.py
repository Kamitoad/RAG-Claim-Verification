"""End-to-end benchmark and re-evaluation using only fake/in-memory components."""

import json
from pathlib import Path

import pytest
import yaml

from rag_claim_verification.config import (
    CorpusConfig,
    OpenAICompatibleConfig,
    load_benchmark_config,
)
from rag_claim_verification.errors import ConfigurationError
from rag_claim_verification.evaluation.benchmark import (
    BenchmarkRunner,
    DefaultComponentFactory,
)
from rag_claim_verification.evaluation.evaluate import evaluate_run
from rag_claim_verification.llm.base import GenerationResult, LLMClient
from rag_claim_verification.retrieval.base import Retriever
from rag_claim_verification.utils.hashing import sha256_file


class DeterministicVerificationClient:
    """Return labels from fixture-specific claim text, never from an external service."""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult:
        del system_prompt
        claim_text = user_prompt.split("Claim:\n", 1)[1].split("\n\nEvidence:", 1)[0]
        citation = ""
        if "1994" in claim_text:
            label = "REFUTED" if "for Ferrari" in claim_text else "SUPPORTED"
            citation = "synthetic_doc_1994"
        elif "Mercedes in 2010" in claim_text:
            label = "SUPPORTED"
            citation = "synthetic_doc_2010"
        elif "McLaren in 1991" in claim_text:
            label = "REFUTED"
            citation = "synthetic_doc_1991"
        else:
            label = "NOT_ENOUGH_EVIDENCE"
        if "BASELINE_WITHOUT_RETRIEVAL" in user_prompt:
            citation = ""
        citations = f'["{citation}"]' if citation else "[]"
        return GenerationResult(
            content=(
                f'{{"label":"{label}","reason":"Deterministic fixture decision.",'
                f'"cited_document_ids":{citations}}}'
            ),
            provider="offline_test",
            requested_model="fake",
            response_model="fake-v1",
            system_fingerprint="offline-fixture-v1",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

    async def close(self) -> None:
        return None


class OfflineFactory:
    """Use the production in-memory retriever and a fake verification model."""

    def __init__(self) -> None:
        self._default = DefaultComponentFactory()

    def create_llm(self, config: OpenAICompatibleConfig) -> LLMClient:
        del config
        return DeterministicVerificationClient()

    def create_retriever(self, config: CorpusConfig) -> Retriever:
        return self._default.create_retriever(config)


class CloseFailingClient(DeterministicVerificationClient):
    async def close(self) -> None:
        raise RuntimeError("synthetic close failure")


class CloseFailingFactory:
    def create_llm(self, config: OpenAICompatibleConfig) -> LLMClient:
        del config
        return CloseFailingClient()

    def create_retriever(self, config: CorpusConfig) -> Retriever:
        raise AssertionError(f"Unexpected retriever request for {config.corpus_id}")


def _write_corpus_config(path: Path, corpus_id: str, manifest: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "corpus_id": corpus_id,
                "manifest_path": str(manifest),
                "retriever": {"type": "in_memory"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_complete_offline_benchmark_and_reevaluation(
    tmp_path: Path, project_root: Path
) -> None:
    clean_config = tmp_path / "clean.yaml"
    noisy_config = tmp_path / "noisy.yaml"
    _write_corpus_config(
        clean_config,
        "clean",
        project_root / "data/manifests/clean_documents.jsonl",
    )
    _write_corpus_config(
        noisy_config,
        "noisy",
        project_root / "data/manifests/noisy_documents.jsonl",
    )
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "offline_integration",
                "claims_file": str(project_root / "data/ground_truth/claims.example.jsonl"),
                "output_directory": str(tmp_path / "runs"),
                "prompts": {
                    "version": "test-v1",
                    "system_path": str(project_root / "prompts/verification_system.txt"),
                    "user_path": str(project_root / "prompts/verification_user.txt"),
                    "repair_path": str(project_root / "prompts/verification_repair.txt"),
                },
                "model_configs": {
                    "model": {
                        "base_url": "http://localhost:1234/v1",
                        "api_key_required": False,
                        "model": "fake",
                        "temperature": 0.0,
                        "max_retries": 0,
                    }
                },
                "conditions": [
                    {"id": "baseline", "mode": "baseline", "model_config": "model"},
                    {
                        "id": "clean_rag",
                        "mode": "rag",
                        "model_config": "model",
                        "corpus_config": str(clean_config),
                        "top_k": 5,
                    },
                    {
                        "id": "noisy_rag",
                        "mode": "rag",
                        "model_config": "model",
                        "corpus_config": str(noisy_config),
                        "top_k": 5,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_benchmark_config(benchmark_path)

    run_directory = await BenchmarkRunner(
        config=config,
        config_path=benchmark_path,
        component_factory=OfflineFactory(),
    ).run()

    expected = {
        "metadata.json",
        "resolved_config.yaml",
        "case_manifest.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "metrics.csv",
        "confusion_matrix.csv",
        "failures.jsonl",
        "summary.md",
    }
    assert expected <= {item.name for item in run_directory.iterdir()}
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["successful_prediction_count"] == 15
    assert metadata["failed_prediction_count"] == 0
    assert metadata["expected_prediction_count"] == 15
    assert metadata["missing_prediction_count"] == 0
    assert metadata["schema_version"] == "2.0"
    assert (run_directory / "inputs/claims.jsonl").is_file()
    assert (run_directory / "inputs/hashes.json").is_file()
    assert sha256_file(run_directory / "inputs/claims.jsonl") == sha256_file(
        project_root / "data/ground_truth/claims.example.jsonl"
    )
    first_metrics = (run_directory / "metrics.json").read_text(encoding="utf-8")

    metrics = evaluate_run(run_directory)

    assert set(metrics["conditions"]) == {"baseline", "clean_rag", "noisy_rag"}
    assert all(
        payload["classification"]["accuracy"] == 1.0 for payload in metrics["conditions"].values()
    )
    assert metrics["completeness"]["complete"] is True
    assert (run_directory / "metrics.json").read_text(encoding="utf-8") == first_metrics

    predictions_path = run_directory / "predictions.jsonl"
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match=r"predictions\.jsonl hash"):
        evaluate_run(run_directory)


@pytest.mark.asyncio
async def test_failed_run_persists_one_explicit_result_for_every_planned_case(
    tmp_path: Path, project_root: Path
) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "checkpoint_integration",
                "claims_file": str(project_root / "data/ground_truth/claims.example.jsonl"),
                "output_directory": str(tmp_path / "runs"),
                "prompts": {
                    "version": "test-v1",
                    "system_path": str(project_root / "prompts/verification_system.txt"),
                    "user_path": str(project_root / "prompts/verification_user.txt"),
                    "repair_path": str(project_root / "prompts/verification_repair.txt"),
                },
                "model_configs": {
                    "model": {
                        "base_url": "http://localhost:1234/v1",
                        "api_key_required": False,
                        "model": "fake",
                        "temperature": 0.0,
                        "max_retries": 0,
                    }
                },
                "conditions": [
                    {"id": "first", "mode": "baseline", "model_config": "model"},
                    {"id": "second", "mode": "baseline", "model_config": "model"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_benchmark_config(benchmark_path)

    with pytest.raises(RuntimeError, match="synthetic close failure"):
        await BenchmarkRunner(
            config=config,
            config_path=benchmark_path,
            component_factory=CloseFailingFactory(),
        ).run()

    run_directory = next((tmp_path / "runs").iterdir())
    predictions = [
        json.loads(line)
        for line in (run_directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert len(predictions) == 10
    assert sum(item["case_status"] == "success" for item in predictions) == 5
    assert sum(item["case_status"] == "pipeline_error" for item in predictions) == 5
    assert metadata["status"] == "failed"
    assert metadata["missing_prediction_count"] == 0
    assert metadata["run_error_type"] == "RuntimeError"
    metrics = evaluate_run(run_directory)
    assert metrics["completeness"]["complete"] is True
    assert metrics["conditions"]["second"]["technical_errors"]["count"] == 5
