"""Model-client interfaces and OpenAI-compatible implementation."""

from rag_claim_verification.llm.base import LLMClient
from rag_claim_verification.llm.openai_compatible import OpenAICompatibleClient

__all__ = ["LLMClient", "OpenAICompatibleClient"]
