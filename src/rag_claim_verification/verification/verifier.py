"""Evidence-grounded verification with one bounded JSON repair attempt."""

import time

from rag_claim_verification.llm.base import LLMClient
from rag_claim_verification.llm.structured_output import (
    StructuredOutputError,
    parse_verification_output,
)
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence
from rag_claim_verification.models.prediction import Prediction, VerificationOutput
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
    ) -> Prediction:
        """Generate, validate, and persist enough detail for later error analysis."""

        if baseline and evidence:
            raise ValueError("baseline verification cannot receive external evidence")
        rendered = self._prompt_builder.build(
            claim,
            evidence,
            baseline=baseline,
            allowed_labels=self._allowed_labels,
        )
        started = time.perf_counter()
        raw_output: str | None = None
        repair_output: str | None = None
        try:
            raw_output = await self._client.generate(
                system_prompt=rendered.system,
                user_prompt=rendered.user,
            )
            try:
                result = self._parse(raw_output, evidence, baseline)
            except StructuredOutputError as first_error:
                repair_prompt = self._repair_prompt(rendered.user, raw_output, str(first_error))
                repair_output = await self._client.generate(
                    system_prompt=rendered.system,
                    user_prompt=repair_prompt,
                )
                try:
                    result = self._parse(repair_output, evidence, baseline)
                except StructuredOutputError as second_error:
                    return self._failed_prediction(
                        claim=claim,
                        condition=condition,
                        evidence=evidence,
                        baseline=baseline,
                        started=started,
                        retrieval_latency_ms=retrieval_latency_ms,
                        retrieval_supports_document_ids=retrieval_supports_document_ids,
                        raw_output=raw_output,
                        repair_output=repair_output,
                        parse_error=f"initial: {first_error}; repair: {second_error}",
                    )
        except Exception as exc:
            return self._failed_prediction(
                claim=claim,
                condition=condition,
                evidence=evidence,
                baseline=baseline,
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_supports_document_ids=retrieval_supports_document_ids,
                raw_output=raw_output,
                repair_output=repair_output,
                error=f"{type(exc).__name__}: {exc}",
            )

        generation_ms = round((time.perf_counter() - started) * 1000)
        return Prediction(
            claim_id=claim.claim_id,
            condition=condition,
            predicted_label=result.label,
            reason=result.reason,
            evidence=evidence,
            cited_document_ids=result.cited_document_ids,
            latency_ms=generation_ms + retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            raw_model_output=raw_output,
            repair_model_output=repair_output,
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
    def _repair_prompt(original_user_prompt: str, invalid: str, error: str) -> str:
        return (
            f"{original_user_prompt}\n\n"
            "The previous response was invalid. Repair only its JSON structure or schema. "
            "Do not add facts, change the evidence, or cite new documents. Return exactly one JSON "
            "object with keys label, reason, and cited_document_ids.\n"
            f"Validation error: {error}\n"
            f"Invalid response:\n{invalid}"
        )

    @staticmethod
    def _failed_prediction(
        *,
        claim: Claim,
        condition: str,
        evidence: list[Evidence],
        baseline: bool,
        started: float,
        retrieval_latency_ms: int,
        retrieval_supports_document_ids: bool,
        raw_output: str | None,
        repair_output: str | None,
        parse_error: str | None = None,
        error: str | None = None,
    ) -> Prediction:
        generation_ms = round((time.perf_counter() - started) * 1000)
        return Prediction(
            claim_id=claim.claim_id,
            condition=condition,
            predicted_label=None,
            reason=None,
            evidence=evidence,
            cited_document_ids=[],
            latency_ms=generation_ms + retrieval_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            raw_model_output=raw_output,
            repair_model_output=repair_output,
            parse_error=parse_error,
            error=error,
            gold_label=claim.gold_label,
            gold_document_ids=claim.gold_document_ids,
            retrieval_supports_document_ids=retrieval_supports_document_ids,
            verification_mode="baseline" if baseline else "rag",
        )
