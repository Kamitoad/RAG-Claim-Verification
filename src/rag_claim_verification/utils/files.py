"""Safe UTF-8 JSON, JSONL, YAML, and text file helpers."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class FileFormatError(ValueError):
    """Raised when a structured input file cannot be decoded."""


def read_text(path: Path) -> str:
    """Read a UTF-8 text file with an actionable decoding error."""

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FileFormatError(f"File is not valid UTF-8: {path}") from exc


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and reject empty or non-mapping documents."""

    try:
        value = yaml.safe_load(read_text(path))
    except yaml.YAMLError as exc:
        raise FileFormatError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FileFormatError(f"Expected a YAML mapping in {path}")
    return value


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read non-blank JSONL records and retain one-based line numbers."""

    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FileFormatError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise FileFormatError(f"Expected a JSON object at {path}:{line_number}")
        records.append((line_number, value))
    return records


def write_json(path: Path, value: Any) -> None:
    """Atomically write pretty, UTF-8 JSON without overwriting through partial writes."""

    serialized = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write_text(path, serialized)


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    """Atomically write JSON Lines records."""

    serialized = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
    )
    atomic_write_text(path, serialized)


def write_yaml(path: Path, value: Any) -> None:
    """Atomically write deterministic, human-readable YAML."""

    serialized = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    atomic_write_text(path, serialized)


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a target atomically after writing a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically preserve exact source bytes for reproducibility snapshots."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def ensure_new_directory(path: Path) -> None:
    """Create a run directory and fail instead of overwriting prior results."""

    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing directory: {path}") from exc
