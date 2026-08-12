"""Configuration consistency tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_claim_verification.config import BenchmarkConfig, load_benchmark_config


def test_tracked_benchmark_configuration_loads(project_root: Path) -> None:
    config = load_benchmark_config(project_root / "configs/benchmark.yaml")

    assert config.conditions[1].corpus_config == project_root / "configs/clean_corpus.yaml"
    assert config.model_configs["default_llm"].temperature == 0.0
    assert config.prompts.repair_path == project_root / "prompts/verification_repair.txt"


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
