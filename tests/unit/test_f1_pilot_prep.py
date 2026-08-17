"""Deterministic tests for the approved Jolpica pilot transformation."""

from pathlib import Path
from typing import Any

import pytest

from rag_claim_verification.models.claim import Claim
from rag_claim_verification.utils.files import read_jsonl
from scripts.prepare_f1_2023_pilot import (
    MAX_CLASSIFICATION_POSITIONS,
    PilotDataError,
    SessionSource,
    SourceDetails,
    derive_claims,
    render_document,
    validate_session_payload,
)


def _session() -> SessionSource:
    return SessionSource(
        source_id="example_race",
        url="https://api.jolpi.ca/ergast/f1/2023/1/results/",
        season="2023",
        round="1",
        expected_race_name="Example Grand Prix",
        session_type="race",
        corpus_role="clean",
        document_id="f1_2023_example_race",
        raw_file="example.json",
        document_file="example.txt",
    )


def _source() -> SourceDetails:
    return SourceDetails(
        name="Jolpica-F1 API",
        base_url="https://api.jolpi.ca/ergast/f1",
        terms_url="https://example.com/terms",
        license="CC BY-NC-SA 4.0",
        license_url="https://example.com/license",
        intended_use="Test",
    )


def _payload() -> dict[str, Any]:
    return {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "season": "2023",
                        "round": "1",
                        "raceName": "Example Grand Prix",
                        "date": "2023-03-05",
                        "Circuit": {"circuitName": "Example Circuit"},
                        "Results": [
                            {
                                "position": "1",
                                "grid": "1",
                                "laps": "57",
                                "status": "Finished",
                                "points": "25",
                                "Driver": {
                                    "givenName": "First",
                                    "familyName": "Driver",
                                },
                                "Constructor": {"name": "Team One"},
                            },
                            {
                                "position": "2",
                                "grid": "3",
                                "laps": "57",
                                "status": "Finished",
                                "points": "18",
                                "Driver": {
                                    "givenName": "Second",
                                    "familyName": "Driver",
                                },
                                "Constructor": {"name": "Team Two"},
                            },
                        ],
                    }
                ]
            }
        }
    }


def test_render_and_claim_derivation_are_deterministic() -> None:
    session = _session()
    race = validate_session_payload(session, _payload())

    rendered = render_document(session, race, _source())
    claims = derive_claims(session, race)

    assert "Position 1: First Driver (Team One); grid 1" in rendered
    assert f"top {MAX_CLASSIFICATION_POSITIONS} positions" in rendered
    assert len(claims) == 6
    assert [claim.gold_label.value for claim in claims] == [
        "SUPPORTED",
        "SUPPORTED",
        "REFUTED",
        "REFUTED",
        "NOT_ENOUGH_EVIDENCE",
        "NOT_ENOUGH_EVIDENCE",
    ]
    assert claims[2].gold_document_ids == [session.document_id]
    assert claims[4].gold_document_ids == []


def test_source_identity_mismatch_is_rejected() -> None:
    payload = _payload()
    payload["MRData"]["RaceTable"]["Races"][0]["round"] = "2"

    with pytest.raises(PilotDataError, match="Source identity mismatch"):
        validate_session_payload(_session(), payload)


def test_gate_claims_are_an_exact_balanced_subset(project_root: Path) -> None:
    full_path = project_root / "data/ground_truth/f1_2023_pilot.jsonl"
    gate_path = project_root / "data/ground_truth/f1_2023_pilot_gate.jsonl"
    full = {
        claim.claim_id: claim
        for _, raw in read_jsonl(full_path)
        for claim in [Claim.model_validate(raw)]
    }
    gate = [Claim.model_validate(raw) for _, raw in read_jsonl(gate_path)]

    assert len(gate) == 6
    assert all(claim == full[claim.claim_id] for claim in gate)
    assert {claim.claim_id[:3] for claim in gate} == {"r01", "r12", "r22"}
    assert sorted(claim.gold_label.value for claim in gate) == [
        "NOT_ENOUGH_EVIDENCE",
        "NOT_ENOUGH_EVIDENCE",
        "REFUTED",
        "REFUTED",
        "SUPPORTED",
        "SUPPORTED",
    ]
