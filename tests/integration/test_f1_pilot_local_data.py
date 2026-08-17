"""Opt-in audit of ignored Jolpica inputs, generated texts, manifests, and gold claims."""

import os
from collections import Counter
from pathlib import Path

import pytest

from rag_claim_verification.ingestion.loader import load_documents
from rag_claim_verification.ingestion.manifest import validate_noisy_superset
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.utils.files import read_jsonl
from rag_claim_verification.utils.hashing import sha256_file
from scripts.prepare_f1_2023_pilot import (
    _load_download_records,
    _text,
    derive_claims,
    load_source_plan,
    parse_json_bytes,
    render_document,
    validate_session_payload,
)


@pytest.mark.local_data
@pytest.mark.skipif(
    os.getenv("RAGCV_RUN_LOCAL_DATA_TESTS") != "1",
    reason="set RAGCV_RUN_LOCAL_DATA_TESTS=1 to audit ignored pilot data",
)
def test_local_pilot_data_and_gold_labels_match_deterministic_derivation(
    project_root: Path,
) -> None:
    plan_path = project_root / "data/sources/f1_2023_pilot_jolpica.yaml"
    raw_directory = project_root / "data/raw/local/f1_2023_pilot"
    clean_manifest = project_root / "data/manifests/local/f1_2023_pilot_v3_clean.jsonl"
    noisy_manifest = project_root / "data/manifests/local/f1_2023_pilot_v3_noisy.jsonl"
    claims_path = project_root / "data/ground_truth/f1_2023_pilot.jsonl"
    plan = load_source_plan(plan_path)
    records = _load_download_records(raw_directory)
    expected_texts: dict[str, str] = {}
    expected_claims: list[Claim] = []

    for session in plan.sessions:
        record = records[session.source_id]
        assert record["url"] == session.url
        assert record["raw_file"] == session.raw_file
        raw_path = raw_directory / session.raw_file
        assert sha256_file(raw_path) == _text(record["sha256"], context="record.sha256")
        race = validate_session_payload(
            session,
            parse_json_bytes(raw_path.read_bytes(), source=session.url),
        )
        expected_texts[session.document_id] = render_document(session, race, plan.source)
        if session.corpus_role == "clean":
            expected_claims.extend(derive_claims(session, race))

    actual_claims = [Claim.model_validate(raw) for _, raw in read_jsonl(claims_path)]
    assert actual_claims == expected_claims
    assert Counter(claim.gold_label for claim in actual_claims) == {
        VerdictLabel.SUPPORTED: 6,
        VerdictLabel.REFUTED: 6,
        VerdictLabel.NOT_ENOUGH_EVIDENCE: 6,
    }

    validate_noisy_superset(clean_manifest, noisy_manifest)
    noisy_documents = load_documents(noisy_manifest)
    assert len(noisy_documents) == 7
    assert {
        document.metadata.document_id: document.text for document in noisy_documents
    } == expected_texts
    combined_text = "\n".join(expected_texts.values()).lower()
    assert all(term not in combined_text for term in ("pit stop", "pit-stop", "tyre", "stint"))
