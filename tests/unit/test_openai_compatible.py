"""OpenAI-compatible client contract tests without network access."""

import json

import httpx
import pytest

from rag_claim_verification.config import OpenAICompatibleConfig
from rag_claim_verification.llm.openai_compatible import OpenAICompatibleClient


@pytest.mark.asyncio
async def test_openai_compatible_client_sends_deterministic_settings() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "local-model-revision",
                "system_fingerprint": "fingerprint-1",
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"message": {"content": '{"label":"SUPPORTED"}'}}],
            },
        )

    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="local-model",
            temperature=0.0,
            seed=17,
            max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.generate(system_prompt="system", user_prompt="user")
    finally:
        await client.close()

    assert result.content == '{"label":"SUPPORTED"}'
    assert result.requested_model == "local-model"
    assert result.response_model == "local-model-revision"
    assert result.system_fingerprint == "fingerprint-1"
    assert result.total_tokens == 14
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 17
    assert captured["response_format"] == {"type": "json_object"}
