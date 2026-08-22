"""Deterministic test-retriever behavior."""

from pathlib import Path

import pytest

from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.models.document import Document
from rag_claim_verification.retrieval.in_memory_retriever import InMemoryKeywordRetriever
from rag_claim_verification.utils.hashing import sha256_text


def _loaded(document_id: str, text: str) -> LoadedDocument:
    path = Path(f"{document_id}.txt")
    return LoadedDocument(
        metadata=Document(
            document_id=document_id,
            title=document_id,
            source="Fixture",
            file_path=path,
        ),
        text=text,
        resolved_file_path=path.resolve(),
        content_hash=sha256_text(text),
    )


@pytest.mark.asyncio
async def test_keyword_retriever_is_deterministic() -> None:
    retriever = InMemoryKeywordRetriever(
        [_loaded("doc_b", "alpha beta"), _loaded("doc_a", "alpha beta")]
    )

    evidence = await retriever.retrieve("alpha", top_k=2)

    assert [item.document_id for item in evidence] == ["doc_a", "doc_b"]
    assert [item.rank for item in evidence] == [1, 2]
    assert all(item.retrieval_score == 1.0 for item in evidence)
