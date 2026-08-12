"""Provider-neutral text-generation interface."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One provider response plus reproducibility metadata exposed by the provider."""

    content: str
    provider: str
    requested_model: str
    response_model: str | None = None
    response_id: str | None = None
    system_fingerprint: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    attempt_count: int = 1


class LLMClient(Protocol):
    """Generate one non-streaming model response."""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Return assistant text and available provider metadata."""

    async def close(self) -> None:
        """Release network resources."""
