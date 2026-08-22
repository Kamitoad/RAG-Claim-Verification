"""Framework-independent retriever contract."""

from typing import Protocol

from rag_claim_verification.models.evidence import Evidence


class Retriever(Protocol):
    """Retrieve ranked evidence without performing claim verification."""

    @property
    def retriever_type(self) -> str:
        """Return a stable identifier recorded in run metadata."""

    @property
    def supports_document_ids(self) -> bool:
        """Indicate whether returned ranks can be mapped to corpus document IDs."""

    async def retrieve(self, query: str, top_k: int) -> list[Evidence]:
        """Return up to top_k evidence passages in rank order."""

    async def close(self) -> None:
        """Release retriever resources."""
