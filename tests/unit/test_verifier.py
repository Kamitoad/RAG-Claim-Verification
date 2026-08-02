"""Verifier repair and failure behavior tests."""

from pathlib import Path

import pytest

from rag_claim_verification.config import PromptConfig
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence
from rag_claim_verification.verification.prompt_builder import PromptBuilder
from rag_claim_verification.verification.verifier import ClaimVerifier


class SequenceClient:
    """Return predefined messages and record bounded call count."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        response = self.responses[self.calls]
        self.calls += 1
        return response

    async def close(self) -> None:
        return None


def _builder(project_root: Path) -> PromptBuilder:
    return PromptBuilder(
        PromptConfig(
            version="test-v1",
            system_path=project_root / "prompts/verification_system.txt",
            user_path=project_root / "prompts/verification_user.txt",
        )
    )


@pytest.mark.asyncio
async def test_verifier_repairs_invalid_json_once(project_root: Path) -> None:
    client = SequenceClient(
        [
            "not json",
            '{"label":"SUPPORTED","reason":"Confirmed.","cited_document_ids":["doc_1"]}',
        ]
    )
    verifier = ClaimVerifier(client, _builder(project_root))

    prediction = await verifier.verify(
        Claim(claim_id="claim_1", claim="A claim", gold_label=VerdictLabel.SUPPORTED),
        [Evidence(document_id="doc_1", text="Evidence", rank=1)],
        condition="clean",
        retrieval_supports_document_ids=True,
    )

    assert client.calls == 2
    assert prediction.predicted_label == VerdictLabel.SUPPORTED
    assert prediction.raw_model_output == "not json"
    assert prediction.repair_model_output is not None


@pytest.mark.asyncio
async def test_verifier_marks_second_parse_failure_without_fallback(project_root: Path) -> None:
    client = SequenceClient(["not json", "still not json"])
    verifier = ClaimVerifier(client, _builder(project_root))

    prediction = await verifier.verify(
        Claim(claim_id="claim_1", claim="A claim"),
        [],
        condition="clean",
    )

    assert client.calls == 2
    assert prediction.predicted_label is None
    assert prediction.parse_error is not None
