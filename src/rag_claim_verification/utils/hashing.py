"""Stable hashing helpers used in run provenance."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest for UTF-8 text."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large local documents remain practical."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_mapping(value: Mapping[str, Any]) -> str:
    """Hash a mapping through deterministic JSON serialization."""

    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


def combine_hashes(*hashes: str) -> str:
    """Combine already-computed hashes in a deterministic order-sensitive digest."""

    return sha256_text("\n".join(hashes))
