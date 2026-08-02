"""Prompt rendering and structured-output tests."""

from pathlib import Path

import pytest

from rag_claim_verification.config import PromptConfig
from rag_claim_verification.llm.structured_output import (
    StructuredOutputError,
    parse_verification_output,
)
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence
from rag_claim_verification.verification.prompt_builder import PromptBuilder


def test_prompt_builder_renders_claim_and_evidence(project_root: Path) -> None:
    builder = PromptBuilder(
        PromptConfig(
            version="test-v1",
            system_path=project_root / "prompts/verification_system.txt",
            user_path=project_root / "prompts/verification_user.txt",
        )
    )
    claim = Claim(claim_id="claim_1", claim="Schumacher drove for Benetton.")
    evidence = [Evidence(document_id="doc_1", text="He drove for Benetton.", rank=1)]

    rendered = builder.build(
        claim,
        evidence,
        baseline=False,
        allowed_labels=tuple(VerdictLabel),
    )

    assert "Schumacher drove for Benetton" in rendered.user
    assert '"document_id": "doc_1"' in rendered.user
    assert "{{claim}}" not in rendered.user


def test_prompt_builder_marks_baseline_without_evidence(project_root: Path) -> None:
    builder = PromptBuilder(
        PromptConfig(
            version="test-v1",
            system_path=project_root / "prompts/verification_system.txt",
            user_path=project_root / "prompts/verification_user.txt",
        )
    )

    rendered = builder.build(
        Claim(claim_id="claim_1", claim="A claim"),
        [],
        baseline=True,
        allowed_labels=tuple(VerdictLabel),
    )

    assert "BASELINE_WITHOUT_RETRIEVAL" in rendered.user
    assert "NO_EXTERNAL_EVIDENCE" in rendered.user


def test_structured_output_is_strict_and_grounded() -> None:
    result = parse_verification_output(
        '{"label":"SUPPORTED","reason":"Confirmed.","cited_document_ids":["doc_1"]}',
        allowed_document_ids={"doc_1"},
        baseline=False,
    )

    assert result.label == VerdictLabel.SUPPORTED


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        '{"label":"SUPPORTED","reason":"x","cited_document_ids":[],"extra":1}',
        '{"label":"MAYBE","reason":"x","cited_document_ids":[]}',
        ('{"label":"SUPPORTED","label":"REFUTED","reason":"x","cited_document_ids":[]}'),
    ],
)
def test_structured_output_rejects_invalid_json_or_schema(raw: str) -> None:
    with pytest.raises(StructuredOutputError):
        parse_verification_output(raw, allowed_document_ids=set(), baseline=False)


def test_structured_output_rejects_unretrieved_citation() -> None:
    raw = '{"label":"SUPPORTED","reason":"x","cited_document_ids":["doc_2"]}'
    with pytest.raises(StructuredOutputError, match="not retrieved"):
        parse_verification_output(raw, allowed_document_ids={"doc_1"}, baseline=False)
