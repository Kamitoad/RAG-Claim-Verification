"""Configuration consistency tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_claim_verification.config import (
    BenchmarkConfig,
    FastEmbedConfig,
    RetrieverConfig,
    load_benchmark_config,
    load_corpus_config,
)


def test_tracked_benchmark_configuration_loads(project_root: Path) -> None:
    config = load_benchmark_config(project_root / "configs/benchmark.yaml")

    assert config.conditions[1].corpus_config == project_root / "configs/clean_corpus.yaml"
    assert config.model_configs["default_llm"].temperature == 0.0
    assert config.prompts.repair_path == project_root / "prompts/verification_repair.txt"


def test_loads_baseline_prompt_diagnostic_as_one_fixed_condition(project_root: Path) -> None:
    config = load_benchmark_config(project_root / "configs/f1_2023_pilot_baseline_diagnostic.yaml")

    assert config.prompts.version == "verification-v3-baseline-knowledge"
    assert config.prompts.system_path == (
        project_root / "prompts/verification_system_v3_baseline_knowledge.txt"
    )
    assert config.claims_file == project_root / "data/ground_truth/f1_2023_pilot_gate.jsonl"
    assert len(config.conditions) == 1
    assert config.conditions[0].id == "llm_baseline_v3"
    assert config.conditions[0].mode == "baseline"
    assert config.conditions[0].corpus_config is None
    assert config.conditions[0].top_k is None


def test_smoke_configuration_records_deterministic_model_settings(project_root: Path) -> None:
    config = load_benchmark_config(project_root / "configs/smoke_benchmark.yaml")

    model = config.model_configs["deterministic_smoke"]
    assert model.temperature == 0.0
    assert model.seed == 17
    assert model.max_retries == 0


def test_comparability_rejects_different_top_k(project_root: Path) -> None:
    raw = load_benchmark_config(project_root / "configs/benchmark.yaml").model_dump(mode="python")
    raw["conditions"][2]["top_k"] = 3

    with pytest.raises(ValidationError, match="same top_k"):
        BenchmarkConfig.model_validate(raw)


def test_fastembed_configuration_rejects_remote_only_fields(tmp_path: Path) -> None:
    config = RetrieverConfig.model_validate(
        {
            "type": "lightrag",
            "working_directory": tmp_path / "index",
            "llm_model_max_async": 1,
            "entity_extract_max_gleaning": 0,
            "max_parallel_insert": 1,
            "lightrag_llm": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_required": False,
                "model": "local-model",
            },
            "embedding": {
                "provider": "fastembed",
                "model": "jinaai/jina-embeddings-v2-small-en",
                "dimension": 512,
                "max_token_size": 8192,
            },
        }
    )

    assert isinstance(config.embedding, FastEmbedConfig)
    assert config.embedding.dimension == 512
    assert config.llm_model_max_async == 1
    assert config.entity_extract_max_gleaning == 0
    assert config.max_parallel_insert == 1

    raw = config.model_dump(mode="python")
    raw["embedding"]["base_url"] = "http://127.0.0.1:11434/v1"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RetrieverConfig.model_validate(raw)


def test_pilot_noisy_config_resolves_declared_base_config(project_root: Path) -> None:
    config = load_corpus_config(project_root / "configs/f1_2023_pilot_noisy.yaml")

    assert config.corpus_id == "f1_2023_pilot_noisy_jolpica_podium_v5"
    assert config.derived_from is not None
    assert config.derived_from.corpus_config == (project_root / "configs/f1_2023_pilot_clean.yaml")
