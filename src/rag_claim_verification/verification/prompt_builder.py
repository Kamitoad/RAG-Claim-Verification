"""Versioned prompt-file loading and deterministic rendering."""

import json
from dataclasses import dataclass
from pathlib import Path

from rag_claim_verification.config import PromptConfig
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.evidence import Evidence
from rag_claim_verification.utils.files import read_text
from rag_claim_verification.utils.hashing import combine_hashes, sha256_file

REQUIRED_USER_PLACEHOLDERS = {
    "{{claim}}",
    "{{evidence}}",
    "{{verification_mode}}",
    "{{allowed_labels}}",
}


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One rendered system/user prompt pair."""

    system: str
    user: str


class PromptBuilder:
    """Load prompts once and render evidence without code-level prompt fragments."""

    def __init__(self, config: PromptConfig) -> None:
        self.version = config.version
        self._system_path = config.system_path
        self._user_path = config.user_path
        self._system = self._load_non_empty(config.system_path)
        self._user_template = self._load_non_empty(config.user_path)
        missing = sorted(
            placeholder
            for placeholder in REQUIRED_USER_PLACEHOLDERS
            if placeholder not in self._user_template
        )
        if missing:
            raise ValueError("User prompt is missing placeholders: " + ", ".join(missing))
        self.prompt_hash = combine_hashes(
            sha256_file(self._system_path), sha256_file(self._user_path)
        )

    @staticmethod
    def _load_non_empty(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
        value = read_text(path).strip()
        if not value:
            raise ValueError(f"Prompt file is empty: {path}")
        return value

    def build(
        self,
        claim: Claim,
        evidence: list[Evidence],
        *,
        baseline: bool,
        allowed_labels: tuple[VerdictLabel, ...],
    ) -> RenderedPrompt:
        """Render a prompt with explicit condition mode and machine-readable evidence."""

        evidence_payload = [
            {
                "document_id": item.document_id,
                "rank": item.rank,
                "source": item.source,
                "publication_date": (
                    item.publication_date.isoformat() if item.publication_date else None
                ),
                "text": item.text,
            }
            for item in evidence
        ]
        replacements = {
            "{{claim}}": claim.claim,
            "{{evidence}}": (
                "NO_EXTERNAL_EVIDENCE"
                if baseline
                else json.dumps(evidence_payload, ensure_ascii=False)
            ),
            "{{verification_mode}}": "BASELINE_WITHOUT_RETRIEVAL" if baseline else "RAG",
            "{{allowed_labels}}": ", ".join(label.value for label in allowed_labels),
        }
        user = self._user_template
        for placeholder, value in replacements.items():
            user = user.replace(placeholder, value)
        return RenderedPrompt(system=self._system, user=user)
