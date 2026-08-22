"""Strict decoding for model-produced verification JSON."""

import json

from pydantic import ValidationError

from rag_claim_verification.models.claim import VerdictLabel
from rag_claim_verification.models.prediction import VerificationOutput


class StructuredOutputError(ValueError):
    """Raised when model text violates the verification output contract."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> object:
    raise StructuredOutputError(f"non-standard JSON constant: {value}")


def parse_verification_output(
    raw_output: str,
    *,
    allowed_document_ids: set[str],
    baseline: bool,
) -> VerificationOutput:
    """Decode one strict JSON object and validate that every citation was retrieved."""

    try:
        raw_value = json.loads(
            raw_output,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON: {exc.msg}") from exc
    try:
        result = VerificationOutput.model_validate(raw_value)
    except ValidationError as exc:
        raise StructuredOutputError(f"schema validation failed: {exc}") from exc
    cited = set(result.cited_document_ids)
    if baseline and cited:
        raise StructuredOutputError("baseline output must not cite external documents")
    if (
        not baseline
        and result.label in {VerdictLabel.SUPPORTED, VerdictLabel.REFUTED}
        and not cited
    ):
        raise StructuredOutputError("decisive RAG output must cite at least one document")
    unknown = sorted(cited - allowed_document_ids)
    if unknown:
        raise StructuredOutputError(
            "output cites documents that were not retrieved: " + ", ".join(unknown)
        )
    return result
