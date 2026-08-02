"""Pinned LightRAG 1.5.4 adapter using only documented public SDK APIs."""

import importlib
import importlib.metadata
import logging
import os
from collections.abc import Awaitable
from typing import Any, cast

from rag_claim_verification.config import LIGHTRAG_VERSION, RetrieverConfig
from rag_claim_verification.errors import ExternalDependencyError
from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.models.document import Document
from rag_claim_verification.models.evidence import Evidence

LOGGER = logging.getLogger(__name__)


class LightRAGAdapter:
    """Encapsulate every version-dependent LightRAG interaction.

    LightRAG 1.5.4 exposes ordered chunks and file paths through ``aquery_data``
    but no documented per-chunk score. The adapter therefore preserves order,
    resolves manifest document IDs, and records scores as ``None``.
    """

    def __init__(self, settings: RetrieverConfig, documents: list[LoadedDocument]) -> None:
        if settings.type != "lightrag":
            raise ValueError("LightRAGAdapter requires retriever type 'lightrag'")
        self._settings = settings
        self._rag: Any = None
        self._query_param_type: Any = None
        self._path_to_document = self._build_path_map(documents)

    @property
    def retriever_type(self) -> str:
        """Return the pinned implementation identifier."""

        return f"lightrag_sdk_{LIGHTRAG_VERSION}"

    @property
    def supports_document_ids(self) -> bool:
        """File-path citations can be deterministically mapped through the manifest."""

        return True

    @staticmethod
    def _build_path_map(documents: list[LoadedDocument]) -> dict[str, Document | None]:
        mapping: dict[str, Document | None] = {}
        for loaded in documents:
            paths = {
                str(loaded.metadata.file_path),
                loaded.metadata.file_path.as_posix(),
                str(loaded.resolved_file_path),
                loaded.resolved_file_path.as_posix(),
                loaded.resolved_file_path.name,
            }
            for path in paths:
                key = LightRAGAdapter._normalize_path(path)
                prior = mapping.get(key)
                if prior is not None and prior.document_id != loaded.metadata.document_id:
                    mapping[key] = None
                elif key not in mapping:
                    mapping[key] = loaded.metadata
        return mapping

    @staticmethod
    def _normalize_path(value: str) -> str:
        return os.path.normcase(value.replace("\\", "/").strip())

    @staticmethod
    def _import_public_api() -> tuple[Any, Any, Any, Any, Any]:
        try:
            installed_version = importlib.metadata.version("lightrag-hku")
            package = importlib.import_module("lightrag")
            openai_module = importlib.import_module("lightrag.llm.openai")
            utils_module = importlib.import_module("lightrag.utils")
        except ImportError as exc:
            raise ExternalDependencyError(
                "LightRAG is not installed. Install the pinned integration with "
                "`pip install -e '.[lightrag]'`."
            ) from exc
        if installed_version != LIGHTRAG_VERSION:
            raise ExternalDependencyError(
                f"Unsupported LightRAG version {installed_version}; expected {LIGHTRAG_VERSION}"
            )
        return (
            package.LightRAG,
            package.QueryParam,
            openai_module.openai_complete_if_cache,
            openai_module.openai_embed,
            utils_module.wrap_embedding_func_with_attrs,
        )

    async def initialize(self) -> None:
        """Create the SDK instance and explicitly initialize all storages."""

        if self._rag is not None:
            return
        settings = self._settings
        if (
            settings.working_directory is None
            or settings.lightrag_llm is None
            or settings.embedding is None
        ):
            raise ValueError("Incomplete LightRAG settings")
        light_rag, query_param, complete, openai_embed, wrap_embedding = self._import_public_api()
        llm_config = settings.lightrag_llm
        embedding_config = settings.embedding
        llm_key = llm_config.api_key(required=llm_config.api_key_required)
        embedding_key = embedding_config.api_key(required=embedding_config.api_key_required)

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs.setdefault("temperature", llm_config.temperature)
            result = complete(
                llm_config.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=llm_key,
                base_url=llm_config.base_url,
                **kwargs,
            )
            return cast(str, await cast(Awaitable[Any], result))

        async def embedding_func(texts: list[str]) -> Any:
            result = openai_embed.func(
                texts,
                model=embedding_config.model,
                api_key=embedding_key,
                base_url=embedding_config.base_url,
            )
            return await cast(Awaitable[Any], result)

        wrapped_embedding = wrap_embedding(
            embedding_dim=embedding_config.dimension,
            max_token_size=embedding_config.max_token_size,
            model_name=embedding_config.model,
        )(embedding_func)
        rag = light_rag(
            working_dir=str(settings.working_directory),
            llm_model_func=llm_model_func,
            llm_model_name=llm_config.model,
            embedding_func=wrapped_embedding,
            chunk_token_size=settings.chunk_token_size,
            chunk_overlap_token_size=settings.chunk_overlap_token_size,
            enable_llm_cache=True,
            default_llm_timeout=max(1, round(llm_config.timeout_seconds)),
            default_embedding_timeout=max(1, round(embedding_config.timeout_seconds)),
        )
        try:
            await cast(Awaitable[None], rag.initialize_storages())
        except Exception:
            try:
                await cast(Awaitable[None], rag.finalize_storages())
            except Exception as cleanup_error:
                LOGGER.warning(
                    "LightRAG cleanup after failed initialization also failed: %s",
                    cleanup_error,
                )
            raise
        self._rag = rag
        self._query_param_type = query_param

    async def ingest(self, documents: list[LoadedDocument]) -> str | None:
        """Insert one validated batch with stable IDs and citation file paths."""

        await self.initialize()
        texts = [document.text for document in documents]
        ids = [document.metadata.document_id for document in documents]
        file_paths = [document.metadata.file_path.as_posix() for document in documents]
        result = await cast(
            Awaitable[Any], self._rag.ainsert(texts, ids=ids, file_paths=file_paths)
        )
        return str(result) if result is not None else None

    async def retrieve(self, query: str, top_k: int) -> list[Evidence]:
        """Retrieve structured LightRAG chunks without invoking its answer generator."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        await self.initialize()
        param = self._query_param_type(
            mode=self._settings.query_mode,
            top_k=top_k,
            chunk_top_k=top_k,
            enable_rerank=self._settings.enable_rerank,
        )
        raw = await cast(Awaitable[Any], self._rag.aquery_data(query, param=param))
        if not isinstance(raw, dict):
            raise ExternalDependencyError("LightRAG aquery_data returned a non-object response")
        if raw.get("status") == "failure":
            return []
        data = raw.get("data", {})
        chunks = data.get("chunks", []) if isinstance(data, dict) else None
        if not isinstance(chunks, list):
            raise ExternalDependencyError("LightRAG retrieval response contains invalid chunks")

        evidence: list[Evidence] = []
        for rank, chunk in enumerate(chunks[:top_k], start=1):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str):
                raise ExternalDependencyError("LightRAG returned a malformed evidence chunk")
            file_path_value = chunk.get("file_path")
            file_path = str(file_path_value) if file_path_value is not None else None
            document = (
                self._path_to_document.get(self._normalize_path(file_path))
                if file_path is not None
                else None
            )
            evidence.append(
                Evidence(
                    document_id=document.document_id if document is not None else None,
                    text=chunk["content"].strip(),
                    rank=rank,
                    retrieval_score=None,
                    source=document.source if document is not None else None,
                    publication_date=(document.publication_date if document is not None else None),
                    file_path=file_path,
                    chunk_id=(
                        str(chunk["chunk_id"]) if chunk.get("chunk_id") is not None else None
                    ),
                )
            )
        return evidence

    async def close(self) -> None:
        """Finalize LightRAG storages on the same event loop used for initialization."""

        if self._rag is not None:
            rag = self._rag
            await cast(Awaitable[None], rag.finalize_storages())
            self._rag = None
