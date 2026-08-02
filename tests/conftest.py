"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the repository root used by tracked examples."""

    return Path(__file__).resolve().parents[1]
