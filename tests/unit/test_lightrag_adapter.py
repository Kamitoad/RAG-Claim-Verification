"""LightRAG public retrieval-response mapping without importing LightRAG."""

from pathlib import Path
from typing import Any

import pytest

from rag_claim_verification.config import (
    OpenAICompatibleConfig,
    OpenAICompatibleEmbeddingConfig,
    RetrieverConfig,
)
from rag_claim_verification.errors import ExternalDependencyError
from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.models.document import Document
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.hashing import sha256_text


class FakeQueryParam:
    def __init__(self, **values: Any) -> None:
        self.values = values


class FakeLightRAG:
    async def aquery_data(self, query: str, *, param: FakeQueryParam) -> dict[str, Any]:
        assert query == "1994 champion"
        assert param.values["chunk_top_k"] == 1
        return {
            "status": "success",
            "data": {
                "chunks": [
                    {
                        "content": "Schumacher won for Benetton.",
                        "file_path": "../corpora/clean/doc.txt",
                        "chunk_id": "chunk-1",
                    }
                ]
            },
        }

    async def finalize_storages(self) -> None:
        return None


class FailedLightRAG:
    async def aquery_data(self, query: str, *, param: FakeQueryParam) -> dict[str, Any]:
        del query, param
        return {"status": "failure", "message": "fixture retrieval failure"}

    async def finalize_storages(self) -> None:
        return None


class FailedFinalizeLightRAG:
    async def finalize_storages(self) -> None:
        raise RuntimeError("storage finalization failed")


@pytest.mark.asyncio
async def test_adapter_maps_public_chunk_path_without_inventing_score(tmp_path: Path) -> None:
    text = "Schumacher won for Benetton."
    loaded = LoadedDocument(
        metadata=Document(
            document_id="doc_1994",
            title="1994",
            source="Fixture",
            file_path=Path("../corpora/clean/doc.txt"),
        ),
        text=text,
        resolved_file_path=(tmp_path / "doc.txt").resolve(),
        content_hash=sha256_text(text),
    )
    settings = RetrieverConfig(
        type="lightrag",
        working_directory=tmp_path / "index",
        lightrag_llm=OpenAICompatibleConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake",
        ),
        embedding=OpenAICompatibleEmbeddingConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake-embedding",
            dimension=3,
        ),
    )
    adapter = LightRAGAdapter(settings, [loaded])
    adapter._rag = FakeLightRAG()
    adapter._query_param_type = FakeQueryParam

    evidence = await adapter.retrieve("1994 champion", top_k=1)
    await adapter.close()

    assert evidence[0].document_id == "doc_1994"
    assert evidence[0].retrieval_score is None
    assert evidence[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_adapter_preserves_lightrag_failure_as_technical_error(tmp_path: Path) -> None:
    settings = RetrieverConfig(
        type="lightrag",
        working_directory=tmp_path / "index",
        lightrag_llm=OpenAICompatibleConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake",
        ),
        embedding=OpenAICompatibleEmbeddingConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake-embedding",
            dimension=3,
        ),
    )
    adapter = LightRAGAdapter(settings, [])
    adapter._rag = FailedLightRAG()
    adapter._query_param_type = FakeQueryParam

    with pytest.raises(ExternalDependencyError, match="fixture retrieval failure"):
        await adapter.retrieve("claim", top_k=1)
    await adapter.close()


@pytest.mark.asyncio
async def test_adapter_releases_shared_data_even_if_storage_finalization_fails(
    tmp_path: Path,
) -> None:
    settings = RetrieverConfig(
        type="lightrag",
        working_directory=tmp_path / "index",
        lightrag_llm=OpenAICompatibleConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake",
        ),
        embedding=OpenAICompatibleEmbeddingConfig(
            base_url="http://localhost:1234/v1",
            api_key_required=False,
            model="fake-embedding",
            dimension=3,
        ),
    )
    released: list[bool] = []
    adapter = LightRAGAdapter(settings, [])
    adapter._rag = FailedFinalizeLightRAG()
    adapter._finalize_shared_data = lambda: released.append(True)

    with pytest.raises(RuntimeError, match="storage finalization failed"):
        await adapter.close()

    assert released == [True]
    assert adapter._rag is None
    assert adapter._finalize_shared_data is None


def test_fastembed_vectors_are_materialized_and_validated() -> None:
    LightRAGAdapter._validate_fastembed_vectors(
        [[1.0, 2.0], [3.0, 4.0]],
        expected_rows=2,
        expected_dimension=2,
    )


def test_fastembed_rejects_wrong_dimension() -> None:
    with pytest.raises(ExternalDependencyError, match="unexpected vector shape"):
        LightRAGAdapter._validate_fastembed_vectors(
            [[1.0, 2.0]],
            expected_rows=1,
            expected_dimension=3,
        )


def test_local_lightrag_defaults_include_placeholder_key_and_seed() -> None:
    config = OpenAICompatibleConfig(
        base_url="http://127.0.0.1:11434/v1",
        api_key_required=False,
        model="local-model",
        temperature=0.0,
        seed=17,
    )
    values: dict[str, Any] = {}

    LightRAGAdapter._apply_llm_defaults(config, values)

    assert LightRAGAdapter._sdk_api_key(None) == "local-no-auth"
    assert values == {"temperature": 0.0, "seed": 17}
