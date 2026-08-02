"""Provider-neutral text-generation interface."""

from typing import Protocol


class LLMClient(Protocol):
    """Generate one non-streaming model response."""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the assistant message text."""

    async def close(self) -> None:
        """Release network resources."""
