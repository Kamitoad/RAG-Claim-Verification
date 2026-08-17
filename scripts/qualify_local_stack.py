"""Qualify Ollama, FastEmbed, LightRAG, retrieval mapping, and verifier JSON locally."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
from fastembed import TextEmbedding

from rag_claim_verification.config import (
    FastEmbedConfig,
    OpenAICompatibleConfig,
    PromptConfig,
    RetrieverConfig,
)
from rag_claim_verification.ingestion.loader import LoadedDocument
from rag_claim_verification.llm.openai_compatible import OpenAICompatibleClient
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.document import Document
from rag_claim_verification.retrieval.lightrag_adapter import LightRAGAdapter
from rag_claim_verification.utils.files import atomic_write_text
from rag_claim_verification.utils.hashing import sha256_text
from rag_claim_verification.verification.prompt_builder import PromptBuilder
from rag_claim_verification.verification.verifier import ClaimVerifier

MODEL_NAME = "ragcv-qwen3-4b-pilot:v1"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


async def _installed_model() -> dict[str, object]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
    payload = response.json()
    models = payload.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags returned an invalid models list")
    for item in models:
        if isinstance(item, dict) and item.get("name") == MODEL_NAME:
            return item
    raise RuntimeError(f"Required Ollama model is not installed: {MODEL_NAME}")


def _loaded_document(directory: Path, document_id: str, text: str) -> LoadedDocument:
    path = directory / f"{document_id}.txt"
    atomic_write_text(path, text)
    return LoadedDocument(
        metadata=Document(
            document_id=document_id,
            title=document_id,
            source="Synthetic local-stack qualification fixture",
            file_path=Path(path.name),
            language="en",
            corpus_tags=["synthetic", "qualification"],
        ),
        text=text,
        resolved_file_path=path.resolve(),
        content_hash=sha256_text(text),
    )


async def qualify(project_root: Path) -> dict[str, object]:
    """Run one bounded end-to-end qualification over two explicit synthetic facts."""

    installed_model = await _installed_model()
    embedding_model = await asyncio.to_thread(TextEmbedding, model_name=EMBEDDING_MODEL)
    vectors = await asyncio.to_thread(
        lambda: list(embedding_model.embed(["Local embedding qualification sentence."]))
    )
    LightRAGAdapter._validate_fastembed_vectors(
        vectors,
        expected_rows=1,
        expected_dimension=512,
    )

    llm_config = OpenAICompatibleConfig(
        base_url=f"{OLLAMA_BASE_URL}/v1",
        api_key_env="RAGCV_LOCAL_UNUSED_API_KEY",
        api_key_required=False,
        model=MODEL_NAME,
        temperature=0.0,
        seed=17,
        timeout_seconds=300,
        max_retries=1,
        request_json_object=True,
    )
    with tempfile.TemporaryDirectory(prefix="ragcv-local-qualification-") as temporary:
        directory = Path(temporary)
        documents = [
            _loaded_document(
                directory,
                "qualification_alpha",
                "In the synthetic qualification fixture, Driver Alpha won Round One.",
            ),
            _loaded_document(
                directory,
                "qualification_beta",
                "In the synthetic qualification fixture, Driver Beta finished second in Round One.",
            ),
        ]
        retriever_config = RetrieverConfig(
            type="lightrag",
            working_directory=directory / "index",
            query_mode="hybrid",
            chunk_token_size=1800,
            chunk_overlap_token_size=150,
            enable_rerank=False,
            llm_model_max_async=1,
            entity_extract_max_gleaning=0,
            max_parallel_insert=1,
            lightrag_llm=llm_config,
            embedding=FastEmbedConfig(
                model=EMBEDDING_MODEL,
                dimension=512,
                max_token_size=8192,
                timeout_seconds=300,
            ),
        )
        adapter = LightRAGAdapter(retriever_config, documents)
        client = OpenAICompatibleClient(llm_config)
        try:
            tracking_id = await adapter.ingest(documents)
            evidence = await adapter.retrieve(
                "Who won Round One in the synthetic qualification fixture?",
                top_k=2,
            )
            verifier = ClaimVerifier(
                client,
                PromptBuilder(
                    PromptConfig(
                        version="verification-v2-citations",
                        system_path=project_root / "prompts/verification_system.txt",
                        user_path=project_root / "prompts/verification_user.txt",
                        repair_path=project_root / "prompts/verification_repair.txt",
                    )
                ),
            )
            prediction = await verifier.verify(
                Claim(
                    claim_id="qualification_claim",
                    claim="Driver Alpha won Round One in the synthetic qualification fixture.",
                ),
                evidence,
                condition="local_stack_qualification",
                retrieval_supports_document_ids=all(
                    item.document_id is not None for item in evidence
                ),
            )
        finally:
            await adapter.close()
            await client.close()

    if prediction.predicted_label != VerdictLabel.SUPPORTED:
        raise RuntimeError(
            "Local verifier qualification did not produce SUPPORTED: "
            f"status={prediction.case_status}, error={prediction.error}, "
            f"parse_error={prediction.parse_error}"
        )
    if "qualification_alpha" not in prediction.cited_document_ids:
        raise RuntimeError(
            "Local verifier qualification did not cite the decisive synthetic document"
        )
    return {
        "status": "qualified",
        "ollama_model": installed_model.get("name"),
        "ollama_digest": installed_model.get("digest"),
        "ollama_size_bytes": installed_model.get("size"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": len(vectors[0]),
        "lightrag_tracking_id": tracking_id,
        "retrieved_document_ids": [item.document_id for item in evidence],
        "predicted_label": prediction.predicted_label.value,
        "cited_document_ids": prediction.cited_document_ids,
        "parse_status": prediction.parse_status.value,
        "model_calls": len(prediction.model_calls),
    }


def main() -> int:
    """Print a machine-readable qualification result and return success only on all gates."""

    project_root = Path(__file__).resolve().parents[1]
    result = asyncio.run(qualify(project_root))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
