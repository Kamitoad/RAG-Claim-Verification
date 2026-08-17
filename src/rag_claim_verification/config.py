"""Validated YAML configuration and environment interpolation."""

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import Field, ValidationError, field_validator, model_validator

from rag_claim_verification.errors import ConfigurationError
from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.utils.files import read_yaml
from rag_claim_verification.utils.hashing import hash_mapping

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
LIGHTRAG_VERSION = "1.5.4"


def _interpolate_string(value: str) -> str:
    """Expand ${NAME} and ${NAME:-default} without ever loading secret values into YAML."""

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        default = match.group(2)
        resolved = os.getenv(variable)
        if resolved is not None and resolved != "":
            return resolved
        if default is not None:
            return default
        raise ConfigurationError(f"Required environment variable is not set: {variable}")

    return ENV_PATTERN.sub(replace, value)


def interpolate_environment(value: Any) -> Any:
    """Recursively interpolate environment references in parsed configuration data."""

    if isinstance(value, str):
        return _interpolate_string(value)
    if isinstance(value, list):
        return [interpolate_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: interpolate_environment(item) for key, item in value.items()}
    return value


class OpenAICompatibleConfig(StrictModel):
    """Settings for an OpenAI-compatible chat-completions endpoint."""

    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key_env: str = "RAGCV_LLM_API_KEY"
    api_key_required: bool = True
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    request_json_object: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require an HTTP endpoint and remove a trailing slash for stable metadata."""

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_environment_name(cls, value: str) -> str:
        """Require an environment-variable name rather than an inline secret."""

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be a valid environment-variable name")
        return value

    def api_key(self, *, required: bool = True) -> str | None:
        """Resolve the API key at call time so it is never serialized in run metadata."""

        value = os.getenv(self.api_key_env)
        if required and not value:
            raise ConfigurationError(
                f"Environment variable {self.api_key_env} is required for provider access"
            )
        return value or None

    def sanitized_endpoint(self) -> str:
        """Return the endpoint without credentials, query parameters, or fragments."""

        parsed = urlsplit(self.base_url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


class OpenAICompatibleEmbeddingConfig(StrictModel):
    """OpenAI-compatible embedding settings required by LightRAG."""

    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key_env: str = "RAGCV_EMBEDDING_API_KEY"
    api_key_required: bool = True
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    max_token_size: int = Field(default=8192, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Require an absolute OpenAI-compatible endpoint URL."""

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_environment_name(cls, value: str) -> str:
        """Validate the indirection used for the embedding secret."""

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be a valid environment-variable name")
        return value

    def api_key(self, *, required: bool = True) -> str | None:
        """Resolve the embedding key only when a provider call is made."""

        value = os.getenv(self.api_key_env)
        if required and not value:
            raise ConfigurationError(
                f"Environment variable {self.api_key_env} is required for embedding access"
            )
        return value or None


class FastEmbedConfig(StrictModel):
    """Local CPU embedding settings backed by FastEmbed."""

    provider: Literal["fastembed"] = "fastembed"
    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    max_token_size: int = Field(default=8192, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)


EmbeddingConfig = Annotated[
    OpenAICompatibleEmbeddingConfig | FastEmbedConfig,
    Field(discriminator="provider"),
]


class RetrieverConfig(StrictModel):
    """Retriever settings for either the real LightRAG adapter or deterministic tests."""

    type: Literal["lightrag", "in_memory"]
    working_directory: Path | None = None
    query_mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"
    chunk_token_size: int = Field(default=1200, gt=0)
    chunk_overlap_token_size: int = Field(default=100, ge=0)
    enable_rerank: bool = False
    llm_model_max_async: int = Field(default=4, ge=1, le=64)
    entity_extract_max_gleaning: int = Field(default=1, ge=0, le=10)
    max_parallel_insert: int = Field(default=3, ge=1, le=64)
    lightrag_llm: OpenAICompatibleConfig | None = None
    embedding: EmbeddingConfig | None = None

    @model_validator(mode="after")
    def validate_type_specific_settings(self) -> "RetrieverConfig":
        """Require provider and storage settings only for a LightRAG retriever."""

        if self.chunk_overlap_token_size >= self.chunk_token_size:
            raise ValueError("chunk_overlap_token_size must be smaller than chunk_token_size")
        if self.type == "lightrag":
            missing: list[str] = []
            if self.working_directory is None:
                missing.append("working_directory")
            if self.lightrag_llm is None:
                missing.append("lightrag_llm")
            if self.embedding is None:
                missing.append("embedding")
            if missing:
                raise ValueError(f"LightRAG retriever requires: {', '.join(missing)}")
        return self


class PromptConfig(StrictModel):
    """Versioned prompt files shared by all experimental conditions."""

    version: str = Field(min_length=1)
    system_path: Path
    user_path: Path
    repair_path: Path


class DerivedIndexConfig(StrictModel):
    """Declare the validated corpus index used as an immutable derivation base."""

    corpus_config: Path


class CorpusConfig(StrictModel):
    """One curated corpus and its retriever/index configuration."""

    corpus_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    manifest_path: Path
    clean_manifest_path: Path | None = None
    derived_from: DerivedIndexConfig | None = None
    retriever: RetrieverConfig
    verification_model: OpenAICompatibleConfig | None = None
    prompts: PromptConfig | None = None

    @model_validator(mode="after")
    def validate_derivation(self) -> "CorpusConfig":
        """Require an explicit clean reference for LightRAG-derived corpora."""

        if self.derived_from is not None:
            if self.retriever.type != "lightrag":
                raise ValueError("derived_from requires retriever.type=lightrag")
            if self.clean_manifest_path is None:
                raise ValueError("derived_from requires clean_manifest_path")
        return self


class BenchmarkCondition(StrictModel):
    """One controlled benchmark condition."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    mode: Literal["baseline", "rag"]
    model_config_id: str = Field(min_length=1, alias="model_config")
    corpus_config: Path | None = None
    top_k: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode(self) -> "BenchmarkCondition":
        """Prevent baseline retrieval and require corpus/top-k for RAG."""

        if self.mode == "baseline":
            if self.corpus_config is not None or self.top_k is not None:
                raise ValueError("baseline conditions cannot define corpus_config or top_k")
        elif self.corpus_config is None or self.top_k is None:
            raise ValueError("rag conditions require corpus_config and top_k")
        return self


class BenchmarkConfig(StrictModel):
    """Multi-condition experiment configuration."""

    experiment_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    claims_file: Path
    output_directory: Path
    prompts: PromptConfig
    model_configs: dict[str, OpenAICompatibleConfig]
    conditions: list[BenchmarkCondition] = Field(min_length=1)
    enforce_comparability: bool = True

    @model_validator(mode="after")
    def validate_experiment(self) -> "BenchmarkConfig":
        """Protect controlled comparisons from accidental model or top-k drift."""

        condition_ids = [condition.id for condition in self.conditions]
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("condition ids must be unique")
        unknown_models = sorted(
            {
                condition.model_config_id
                for condition in self.conditions
                if condition.model_config_id not in self.model_configs
            }
        )
        if unknown_models:
            raise ValueError(f"unknown model_config references: {', '.join(unknown_models)}")
        if self.enforce_comparability:
            referenced = [self.model_configs[item.model_config_id] for item in self.conditions]
            first = referenced[0].model_dump(mode="json")
            if any(model.model_dump(mode="json") != first for model in referenced[1:]):
                raise ValueError(
                    "all conditions must use identical model settings when comparability "
                    "is enforced"
                )
            rag_top_k = {item.top_k for item in self.conditions if item.mode == "rag"}
            if len(rag_top_k) > 1:
                raise ValueError(
                    "all RAG conditions must use the same top_k when comparability is enforced"
                )
        return self


def _resolve_path(path: Path, config_path: Path) -> Path:
    """Resolve a path against the directory containing its declaring config."""

    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _load_config_data(path: Path) -> dict[str, Any]:
    """Load dotenv once, then parse and interpolate a YAML mapping."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {resolved}")
    load_dotenv(override=False)
    value = interpolate_environment(read_yaml(resolved))
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a mapping in configuration: {resolved}")
    return value


def load_corpus_config(path: Path) -> CorpusConfig:
    """Load a corpus config and resolve all declared paths."""

    resolved = path.resolve()
    try:
        config = CorpusConfig.model_validate(_load_config_data(resolved))
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid corpus configuration {resolved}:\n{exc}") from exc
    retriever = config.retriever
    if retriever.working_directory is not None:
        retriever = retriever.model_copy(
            update={"working_directory": _resolve_path(retriever.working_directory, resolved)}
        )
    prompts = config.prompts
    if prompts is not None:
        prompts = prompts.model_copy(
            update={
                "system_path": _resolve_path(prompts.system_path, resolved),
                "user_path": _resolve_path(prompts.user_path, resolved),
                "repair_path": _resolve_path(prompts.repair_path, resolved),
            }
        )
    derived_from = config.derived_from
    if derived_from is not None:
        derived_from = derived_from.model_copy(
            update={
                "corpus_config": _resolve_path(derived_from.corpus_config, resolved),
            }
        )
    return config.model_copy(
        update={
            "manifest_path": _resolve_path(config.manifest_path, resolved),
            "clean_manifest_path": (
                _resolve_path(config.clean_manifest_path, resolved)
                if config.clean_manifest_path is not None
                else None
            ),
            "derived_from": derived_from,
            "retriever": retriever,
            "prompts": prompts,
        }
    )


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark config and resolve all config-relative paths."""

    resolved = path.resolve()
    try:
        config = BenchmarkConfig.model_validate(_load_config_data(resolved))
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid benchmark configuration {resolved}:\n{exc}") from exc
    prompts = config.prompts.model_copy(
        update={
            "system_path": _resolve_path(config.prompts.system_path, resolved),
            "user_path": _resolve_path(config.prompts.user_path, resolved),
            "repair_path": _resolve_path(config.prompts.repair_path, resolved),
        }
    )
    conditions = [
        condition.model_copy(
            update={
                "corpus_config": (
                    _resolve_path(condition.corpus_config, resolved)
                    if condition.corpus_config is not None
                    else None
                )
            }
        )
        for condition in config.conditions
    ]
    return config.model_copy(
        update={
            "claims_file": _resolve_path(config.claims_file, resolved),
            "output_directory": _resolve_path(config.output_directory, resolved),
            "prompts": prompts,
            "conditions": conditions,
        }
    )


def corpus_index_config_hash(config: CorpusConfig) -> str:
    """Hash corpus identity and index settings, excluding location-only paths."""

    retriever = config.retriever.model_dump(mode="json")
    retriever.pop("working_directory", None)
    return hash_mapping(
        {
            "corpus_id": config.corpus_id,
            "retriever": retriever,
        }
    )


def retriever_index_config_hash(config: CorpusConfig) -> str:
    """Hash only index-producing retriever settings, excluding its storage location."""

    retriever = config.retriever.model_dump(mode="json")
    retriever.pop("working_directory", None)
    return hash_mapping(retriever)
