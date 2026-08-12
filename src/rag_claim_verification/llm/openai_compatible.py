"""Minimal OpenAI-compatible chat-completions client."""

import asyncio
import logging
from typing import Any

import httpx

from rag_claim_verification.config import OpenAICompatibleConfig
from rag_claim_verification.errors import ProviderError
from rag_claim_verification.llm.base import GenerationResult

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {408, 409, 429}


class OpenAICompatibleClient:
    """Call external APIs or local servers through the common chat-completions contract."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        api_key = config.api_key(required=config.api_key_required)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=f"{config.base_url}/",
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )

    async def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Generate text with bounded retries for transient transport/provider failures."""

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._config.temperature,
            "stream": False,
        }
        if self._config.request_json_object:
            payload["response_format"] = {"type": "json_object"}
        if self._config.seed is not None:
            payload["seed"] = self._config.seed

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._client.post("chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS_CODES or response.status_code >= 500:
                    raise ProviderError(
                        f"Provider returned retryable HTTP status {response.status_code}"
                    )
                if response.is_error:
                    body = response.text[:500].replace("\n", " ")
                    raise ProviderError(
                        f"Provider returned HTTP status {response.status_code}: {body}"
                    )
                return self._extract_result(response, attempt_count=attempt + 1)
            except (httpx.TransportError, ProviderError) as exc:
                last_error = exc
                if attempt >= self._config.max_retries or (
                    isinstance(exc, ProviderError) and "retryable" not in str(exc)
                ):
                    break
                delay = min(0.5 * (2**attempt), 4.0)
                LOGGER.warning(
                    "Model request failed transiently; retrying (%d/%d)",
                    attempt + 1,
                    self._config.max_retries,
                )
                await asyncio.sleep(delay)
        raise ProviderError(f"Model request failed: {last_error}") from last_error

    def _extract_result(self, response: httpx.Response, *, attempt_count: int) -> GenerationResult:
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider response does not match chat-completions schema") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Provider returned an empty assistant message")
        usage = payload.get("usage")
        usage_mapping = usage if isinstance(usage, dict) else {}

        def optional_string(key: str) -> str | None:
            value = payload.get(key)
            return value if isinstance(value, str) and value else None

        def optional_token_count(key: str) -> int | None:
            value = usage_mapping.get(key)
            return value if isinstance(value, int) and value >= 0 else None

        return GenerationResult(
            content=content,
            provider="openai_compatible",
            requested_model=self._config.model,
            response_model=optional_string("model"),
            response_id=optional_string("id"),
            system_fingerprint=optional_string("system_fingerprint"),
            prompt_tokens=optional_token_count("prompt_tokens"),
            completion_tokens=optional_token_count("completion_tokens"),
            total_tokens=optional_token_count("total_tokens"),
            attempt_count=attempt_count,
        )

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""

        await self._client.aclose()
