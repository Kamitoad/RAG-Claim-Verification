"""Deterministic keyword retriever for tests and offline demonstrations."""

import re

from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.models.evidence import Evidence

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(value)}


class InMemoryKeywordRetriever:
    """Rank whole documents by deterministic query-token coverage."""

    def __init__(self, documents: list[LoadedDocument]) -> None:
        self._documents = tuple(documents)

    @property
    def retriever_type(self) -> str:
        """Return the stable adapter name."""

        return "in_memory_keyword"

    @property
    def supports_document_ids(self) -> bool:
        """The manifest supplies an exact ID for every returned document."""

        return True

    async def retrieve(self, query: str, top_k: int) -> list[Evidence]:
        """Return documents with at least one query-token match."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        scored: list[tuple[float, LoadedDocument]] = []
        for document in self._documents:
            searchable = " ".join(
                filter(
                    None,
                    [
                        document.metadata.title,
                        document.metadata.topic,
                        document.text,
                    ],
                )
            )
            score = len(query_tokens & _tokens(searchable)) / len(query_tokens)
            if score > 0.0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].metadata.document_id))
        return [
            Evidence(
                document_id=document.metadata.document_id,
                text=document.text.strip(),
                rank=rank,
                retrieval_score=score,
                source=document.metadata.source,
                publication_date=document.metadata.publication_date,
                file_path=document.metadata.file_path.as_posix(),
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]

    async def close(self) -> None:
        """The in-memory retriever owns no external resources."""
