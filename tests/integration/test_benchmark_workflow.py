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
from rag_claim_verification.evaluation.benchmark import (
    BenchmarkRunner,
    DefaultComponentFactory,
)
from rag_claim_verification.evaluation.evaluate import evaluate_run
from rag_claim_verification.llm.base import LLMClient
from rag_claim_verification.retrieval.base import Retriever


class DeterministicVerificationClient:
    """Return labels from fixture-specific claim text, never from an external service."""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
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
        return (
            f'{{"label":"{label}","reason":"Deterministic fixture decision.",'
            f'"cited_document_ids":{citations}}}'
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
    first_metrics = (run_directory / "metrics.json").read_text(encoding="utf-8")

    metrics = evaluate_run(run_directory)

    assert set(metrics["conditions"]) == {"baseline", "clean_rag", "noisy_rag"}
    assert all(
        payload["classification"]["accuracy"] == 1.0 for payload in metrics["conditions"].values()
    )
    assert (run_directory / "metrics.json").read_text(encoding="utf-8") == first_metrics
