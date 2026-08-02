"""Baseline verification without retrieval."""

from rag_claim_verification.models.claim import Claim
from rag_claim_verification.models.prediction import Prediction
from rag_claim_verification.verification.verifier import ClaimVerifier


class BaselineVerifier:
    """Make the no-retrieval experimental condition explicit in code and output."""

    def __init__(self, verifier: ClaimVerifier) -> None:
        self._verifier = verifier

    async def verify(self, claim: Claim, *, condition: str) -> Prediction:
        """Verify a claim with no external evidence or document citations."""

        return await self._verifier.verify(
            claim,
            [],
            condition=condition,
            baseline=True,
            retrieval_supports_document_ids=False,
        )
