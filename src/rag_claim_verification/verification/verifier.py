"""Evidence-grounded verification with one bounded JSON repair attempt."""

import time
from typing import Literal

from rag_claim_verification.llm.base import GenerationResult, LLMClient
from rag_claim_verification.llm.structured_output import (
    StructuredOutputError,
    parse_verification_output,
)
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence
from rag_claim_verification.models.prediction import (
    CaseStatus,
    ModelCallMetadata,
    ParseStatus,
    Prediction,
    RetrievalStatus,
    VerificationOutput,
)
from rag_claim_verification.verification.prompt_builder import PromptBuilder

DEFAULT_LABELS = tuple(VerdictLabel)


class ClaimVerifier:
    """Turn a claim and already-retrieved evidence into a validated prediction."""

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: PromptBuilder,
        *,
        allowed_labels: tuple[VerdictLabel, ...] = DEFAULT_LABELS,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder
        self._allowed_labels = allowed_labels

    async def verify(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        condition: str,
        baseline: bool = False,
        retrieval_latency_ms: int = 0,
        retrieval_supports_document_ids: bool = False,
        retrieval_status: RetrievalStatus | None = None,
    ) -> Prediction:
        """Generate, validate, and persist enough detail for later error analysis."""

        started = time.perf_counter()
        if baseline and evidence:
            raise ValueError("baseline verification cannot receive external evidence")
        observed_retrieval_status = retrieval_status or (
            RetrievalStatus.NOT_APPLICABLE
            if baseline
            else RetrievalStatus.SUCCESS
            if evidence
            else RetrievalStatus.SUCCESS_EMPTY
        )
        raw_output: str | None = None
        repair_output: str | None = None
        initial_parse_error: str | None = None
        generation_latency_ms = 0
        repair_latency_ms = 0
        generation_started: float | None = None
        model_calls: list[ModelCallMetadata] = []
        stage = "prompt"
        parse_status = ParseStatus.NOT_STARTED
        try:
            rendered = self._prompt_builder.build(
                claim,
                evidence,
                baseline=baseline,
                allowed_labels=self._allowed_labels,
            )
            stage = "model"
            generation_started = time.perf_counter()
            initial_result = await self._client.generate(
                system_prompt=rendered.system,
                user_prompt=rendered.user,
            )
            generation_latency_ms = round((time.perf_counter() - generation_started) * 1000)
            raw_output = initial_result.content
            model_calls.append(
                self._model_call_metadata(
                    initial_result,
                    purpose="initial",
                    latency_ms=generation_latency_ms,
                )
            )
            try:
                result = self._parse(raw_output, evidence, baseline)
                parse_status = ParseStatus.VALID_FIRST_PASS
            except StructuredOutputError as first_error:
                initial_parse_error = str(first_error)
                repair_prompt = self._prompt_builder.build_repair(
                    original_user_prompt=rendered.user,
                    invalid_output=raw_output,
                    validation_error=initial_parse_error,
                )
                repair_started = time.perf_counter()
                try:
                    repair_result = await self._client.generate(
                        system_prompt=rendered.system,
                        user_prompt=repair_prompt,
                    )
                except Exception:
                    repair_latency_ms = round((time.perf_counter() - repair_started) * 1000)
                    parse_status = ParseStatus.REPAIR_CALL_FAILED
                    raise
                repair_latency_ms = round((time.perf_counter() - repair_started) * 1000)
                repair_output = repair_result.content
                model_calls.append(
                    self._model_call_metadata(
                        repair_result,
                        purpose="repair",
                        latency_ms=repair_latency_ms,
                    )
                )
                try:
                    result = self._parse(repair_output, evidence, baseline)
                    parse_status = ParseStatus.VALID_AFTER_REPAIR
                except StructuredOutputError as second_error:
                    return self._failed_prediction(
                        claim=claim,
                        condition=condition,
                        evidence=evidence,
                        baseline=baseline,
                        started=started,
                        retrieval_latency_ms=retrieval_latency_ms,
                        generation_latency_ms=generation_latency_ms,
                        repair_latency_ms=repair_latency_ms,
                        retrieval_supports_document_ids=retrieval_supports_document_ids,
                        retrieval_status=observed_retrieval_status,
                        parse_status=ParseStatus.INVALID_AFTER_REPAIR,
                        model_calls=model_calls,
                        raw_output=raw_output,
                        repair_output=repair_output,
                        initial_parse_error=initial_parse_error,
                        parse_error=f"initial: {first_error}; repair: {second_error}",
                        case_status=CaseStatus.PARSE_ERROR,
                    )
        except Exception as exc:
            if raw_output is None and generation_started is not None:
                generation_latency_ms = round((time.perf_counter() - generation_started) * 1000)
            return self._failed_prediction(
                claim=claim,
                condition=condition,
                evidence=evidence,
                baseline=baseline,
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                repair_latency_ms=repair_latency_ms,
                retrieval_supports_document_ids=retrieval_supports_document_ids,
                retrieval_status=observed_retrieval_status,
                parse_status=parse_status,
                model_calls=model_calls,
                raw_output=raw_output,
                repair_output=repair_output,
                initial_parse_error=initial_parse_error,
                error=f"{type(exc).__name__}: {exc}",
                error_stage=stage,
                error_type=type(exc).__name__,
                case_status=(
                    CaseStatus.MODEL_ERROR if stage == "model" else CaseStatus.PIPELINE_ERROR
                ),
            )

        verification_ms = round((time.perf_counter() - started) * 1000)
        return Prediction(
            case_id=self._case_id(condition, claim.claim_id),
            claim_id=claim.claim_id,
            claim=claim.claim,
            condition=condition,
            case_status=CaseStatus.SUCCESS,
            retrieval_status=observed_retrieval_status,
            parse_status=parse_status,
            predicted_label=result.label,
            reason=result.reason,
            evidence=evidence,
            cited_document_ids=result.cited_document_ids,
            latency_ms=verification_ms + retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            repair_latency_ms=repair_latency_ms,
            model_calls=model_calls,
            raw_model_output=raw_output,
            repair_model_output=repair_output,
            initial_parse_error=initial_parse_error,
            parse_error=None,
            error=None,
            gold_label=claim.gold_label,
            gold_document_ids=claim.gold_document_ids,
            retrieval_supports_document_ids=retrieval_supports_document_ids,
            verification_mode="baseline" if baseline else "rag",
        )

    @staticmethod
    def _parse(raw_output: str, evidence: list[Evidence], baseline: bool) -> VerificationOutput:
        allowed_ids = {item.document_id for item in evidence if item.document_id is not None}
        return parse_verification_output(
            raw_output,
            allowed_document_ids=allowed_ids,
            baseline=baseline,
        )

    @staticmethod
    def _model_call_metadata(
        result: GenerationResult,
        *,
        purpose: Literal["initial", "repair"],
        latency_ms: int,
    ) -> ModelCallMetadata:
        return ModelCallMetadata(
            purpose=purpose,
            provider=result.provider,
            requested_model=result.requested_model,
            response_model=result.response_model,
            response_id=result.response_id,
            system_fingerprint=result.system_fingerprint,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            attempt_count=result.attempt_count,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _case_id(condition: str, claim_id: str) -> str:
        return f"{condition}:{claim_id}"

    @staticmethod
    def _failed_prediction(
        *,
        claim: Claim,
        condition: str,
        evidence: list[Evidence],
        baseline: bool,
        started: float,
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        repair_latency_ms: int,
        retrieval_supports_document_ids: bool,
        retrieval_status: RetrievalStatus,
        parse_status: ParseStatus,
        model_calls: list[ModelCallMetadata],
        raw_output: str | None,
        repair_output: str | None,
        initial_parse_error: str | None,
        parse_error: str | None = None,
        error: str | None = None,
        error_stage: str | None = None,
        error_type: str | None = None,
        case_status: CaseStatus,
    ) -> Prediction:
        verification_ms = round((time.perf_counter() - started) * 1000)
        return Prediction(
            case_id=ClaimVerifier._case_id(condition, claim.claim_id),
            claim_id=claim.claim_id,
            claim=claim.claim,
            condition=condition,
            case_status=case_status,
            retrieval_status=retrieval_status,
            parse_status=parse_status,
            predicted_label=None,
            reason=None,
            evidence=evidence,
            cited_document_ids=[],
            latency_ms=verification_ms + retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            repair_latency_ms=repair_latency_ms,
            model_calls=model_calls,
            raw_model_output=raw_output,
            repair_model_output=repair_output,
            initial_parse_error=initial_parse_error,
            parse_error=parse_error,
            error=error,
            error_stage=error_stage,
            error_type=error_type,
            gold_label=claim.gold_label,
            gold_document_ids=claim.gold_document_ids,
            retrieval_supports_document_ids=retrieval_supports_document_ids,
            verification_mode="baseline" if baseline else "rag",
        )
