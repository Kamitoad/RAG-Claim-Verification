"""Opt-in check for the pinned external LightRAG dependency."""

import importlib.metadata
import os

import pytest

from rag_claim_verification.config import LIGHTRAG_VERSION


@pytest.mark.external
@pytest.mark.skipif(
    os.getenv("RAGCV_RUN_EXTERNAL_TESTS") != "1",
    reason="set RAGCV_RUN_EXTERNAL_TESTS=1 after installing the LightRAG extra",
)
def test_installed_lightrag_version_matches_adapter_pin() -> None:
    assert importlib.metadata.version("lightrag-hku") == LIGHTRAG_VERSION
