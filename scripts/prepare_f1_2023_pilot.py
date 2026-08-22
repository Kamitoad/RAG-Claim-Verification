"""Download and deterministically prepare the approved local F1 2023 pilot corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from rag_claim_verification.models.base import StrictModel
from rag_claim_verification.models.claim import Claim, VerdictLabel
from rag_claim_verification.models.document import Document
from rag_claim_verification.utils.files import (
    atomic_write_bytes,
    atomic_write_text,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
)
from rag_claim_verification.utils.hashing import sha256_file, sha256_text

TRANSFORMER_VERSION = "f1-2023-pilot-jolpica-v3-podium"
MAX_CLASSIFICATION_POSITIONS = 3
USER_AGENT = "RAG-Claim-Verification university research pilot/0.1"


class PilotDataError(ValueError):
    """Raised when a source plan or downloaded response violates the pilot contract."""


class SourceDetails(StrictModel):
    """License and attribution data shared by every approved endpoint."""

    name: str = Field(min_length=1)
    base_url: str
    terms_url: str
    license: str = Field(min_length=1)
    license_url: str
    intended_use: str = Field(min_length=1)

    @field_validator("base_url", "terms_url", "license_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require explicit HTTPS source and license URLs."""

        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("source URLs must be absolute HTTPS URLs")
        return value


class SessionSource(StrictModel):
    """One pinned Jolpica response and its deterministic corpus role."""

    source_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    url: str
    season: str = Field(pattern=r"^\d{4}$")
    round: str = Field(pattern=r"^[1-9]\d*$")
    expected_race_name: str = Field(min_length=1)
    session_type: Literal["race", "qualifying", "sprint"]
    corpus_role: Literal["clean", "noise"]
    document_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    raw_file: str = Field(min_length=1)
    document_file: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        """Restrict acquisition to the declared Jolpica API host."""

        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.hostname != "api.jolpi.ca":
            raise ValueError("session URL must use https://api.jolpi.ca")
        return value

    @field_validator("raw_file", "document_file")
    @classmethod
    def validate_leaf_file(cls, value: str) -> str:
        """Prevent a source declaration from escaping its local target directory."""

        path = Path(value)
        if path.name != value or value in {".", ".."}:
            raise ValueError("output names must be simple leaf filenames")
        return value


class SourcePlan(StrictModel):
    """Strict, reviewable acquisition plan for the complete pilot download."""

    schema_version: Literal[1]
    source: SourceDetails
    sessions: list[SessionSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sessions(self) -> SourcePlan:
        """Reject collisions before any network or filesystem operation."""

        for field_name in ("source_id", "url", "document_id", "raw_file", "document_file"):
            values = [getattr(session, field_name) for session in self.sessions]
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate session {field_name}")
        if not any(item.corpus_role == "clean" for item in self.sessions):
            raise ValueError("source plan must contain at least one clean document")
        return self


def load_source_plan(path: Path) -> SourcePlan:
    """Load and strictly validate the reviewed YAML source plan."""

    try:
        return SourcePlan.model_validate(read_yaml(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise PilotDataError(f"Invalid source plan {path}: {exc}") from exc


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PilotDataError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise PilotDataError(f"non-standard JSON constant: {value}")


def parse_json_bytes(content: bytes, *, source: str) -> dict[str, Any]:
    """Decode one standards-compliant JSON object while preserving raw bytes separately."""

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotDataError(f"Response is not UTF-8 for {source}") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise PilotDataError(f"Response is not valid JSON for {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise PilotDataError(f"Expected a JSON object for {source}")
    return value


def _object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PilotDataError(f"Expected an object at {context}")
    return value


def _list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PilotDataError(f"Expected a list at {context}")
    return value


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotDataError(f"Expected non-empty text at {context}")
    return value.strip()


def validate_session_payload(session: SessionSource, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate source identity and return the single requested race object."""

    mr_data = _object(payload.get("MRData"), context="MRData")
    race_table = _object(mr_data.get("RaceTable"), context="MRData.RaceTable")
    races = _list(race_table.get("Races"), context="MRData.RaceTable.Races")
    if len(races) != 1:
        raise PilotDataError(
            f"{session.source_id} must contain exactly one race, received {len(races)}"
        )
    race = _object(races[0], context=f"{session.source_id}.race")
    observed = {
        "season": _text(race.get("season"), context="race.season"),
        "round": _text(race.get("round"), context="race.round"),
        "raceName": _text(race.get("raceName"), context="race.raceName"),
    }
    expected = {
        "season": session.season,
        "round": session.round,
        "raceName": session.expected_race_name,
    }
    if observed != expected:
        raise PilotDataError(
            f"Source identity mismatch for {session.source_id}: "
            f"observed {observed}, expected {expected}"
        )
    collection_name = {
        "race": "Results",
        "qualifying": "QualifyingResults",
        "sprint": "SprintResults",
    }[session.session_type]
    rows = _list(race.get(collection_name), context=f"race.{collection_name}")
    if not rows:
        raise PilotDataError(f"{session.source_id} contains no {collection_name}")
    return race


async def download_sources(plan_path: Path, raw_directory: Path) -> None:
    """Download the complete approved plan and then atomically persist every raw response."""

    plan = load_source_plan(plan_path)
    if raw_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw directory: {raw_directory}")

    downloaded: list[tuple[SessionSource, bytes]] = []
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    ) as client:
        for index, session in enumerate(plan.sessions):
            response = await client.get(session.url)
            response.raise_for_status()
            content = response.content
            validate_session_payload(
                session,
                parse_json_bytes(content, source=session.url),
            )
            downloaded.append((session, content))
            if index + 1 < len(plan.sessions):
                await asyncio.sleep(0.3)

    raw_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    retrieved_at = datetime.now(UTC).isoformat()
    for session, content in downloaded:
        target = raw_directory / session.raw_file
        atomic_write_bytes(target, content)
        records.append(
            {
                "source_id": session.source_id,
                "url": session.url,
                "raw_file": session.raw_file,
                "sha256": sha256_file(target),
                "byte_count": len(content),
            }
        )
    write_json(
        raw_directory / "download_manifest.json",
        {
            "schema_version": 1,
            "retrieved_at": retrieved_at,
            "source_plan": plan_path.name,
            "source_plan_sha256": sha256_file(plan_path),
            "transformer_version": TRANSFORMER_VERSION,
            "records": records,
        },
    )


def _driver_name(row: dict[str, Any], *, context: str) -> str:
    driver = _object(row.get("Driver"), context=f"{context}.Driver")
    given = _text(driver.get("givenName"), context=f"{context}.Driver.givenName")
    family = _text(driver.get("familyName"), context=f"{context}.Driver.familyName")
    return f"{given} {family}"


def _common_header(
    session: SessionSource,
    race: dict[str, Any],
    source: SourceDetails,
) -> list[str]:
    circuit = _object(race.get("Circuit"), context="race.Circuit")
    circuit_name = _text(circuit.get("circuitName"), context="race.Circuit.circuitName")
    event_date = _text(race.get("date"), context="race.date")
    title_type = {"race": "Race", "qualifying": "Qualifying", "sprint": "Sprint"}[
        session.session_type
    ]
    return [
        f"2023 {session.expected_race_name} — {title_type} classification",
        "",
        f"Season: {session.season}",
        f"Round: {session.round}",
        f"Event: {session.expected_race_name}",
        f"Session: {title_type}",
        f"Circuit: {circuit_name}",
        f"Event date: {event_date}",
        f"Source: {source.name}",
        f"Source URL: {session.url}",
        f"Data license: {source.license}",
        "",
        f"Classification (top {MAX_CLASSIFICATION_POSITIONS} positions as returned):",
    ]


def render_document(
    session: SessionSource,
    race: dict[str, Any],
    source: SourceDetails,
) -> str:
    """Render selected structured facts into stable, self-authored English prose."""

    lines = _common_header(session, race, source)
    collection_name = {
        "race": "Results",
        "qualifying": "QualifyingResults",
        "sprint": "SprintResults",
    }[session.session_type]
    rows = _list(race.get(collection_name), context=f"race.{collection_name}")
    rows = rows[:MAX_CLASSIFICATION_POSITIONS]
    for index, raw_row in enumerate(rows, start=1):
        row = _object(raw_row, context=f"{collection_name}[{index}]")
        position = _text(row.get("position"), context=f"{collection_name}[{index}].position")
        driver = _driver_name(row, context=f"{collection_name}[{index}]")
        constructor = _object(
            row.get("Constructor"),
            context=f"{collection_name}[{index}].Constructor",
        )
        constructor_name = _text(
            constructor.get("name"),
            context=f"{collection_name}[{index}].Constructor.name",
        )
        facts = [f"Position {position}: {driver} ({constructor_name})"]
        if session.session_type == "qualifying":
            for segment in ("Q1", "Q2", "Q3"):
                value = row.get(segment)
                if isinstance(value, str) and value.strip():
                    facts.append(f"{segment} {value.strip()}")
        else:
            for key, label in (
                ("grid", "grid"),
                ("laps", "laps"),
                ("status", "status"),
                ("points", "points"),
            ):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    facts.append(f"{label} {value.strip()}")
            time_value = row.get("Time")
            if isinstance(time_value, dict) and isinstance(time_value.get("time"), str):
                facts.append(f"time {time_value['time'].strip()}")
            fastest = row.get("FastestLap")
            if isinstance(fastest, dict):
                rank = fastest.get("rank")
                lap = fastest.get("lap")
                lap_time = fastest.get("Time")
                if isinstance(rank, str) and isinstance(lap, str):
                    facts.append(f"fastest-lap rank {rank} on lap {lap}")
                if isinstance(lap_time, dict) and isinstance(lap_time.get("time"), str):
                    facts.append(f"fastest-lap time {lap_time['time'].strip()}")
        lines.append("; ".join(facts) + ".")
    return "\n".join(lines) + "\n"


def derive_claims(session: SessionSource, race: dict[str, Any]) -> list[Claim]:
    """Create two claims per label from one clean race response."""

    if session.session_type != "race" or session.corpus_role != "clean":
        raise PilotDataError("Claims can only be derived from clean race sessions")
    rows = _list(race.get("Results"), context="race.Results")
    if len(rows) < 2:
        raise PilotDataError(f"{session.source_id} needs at least two classified drivers")
    winner_row = _object(rows[0], context="race.Results[1]")
    second_row = _object(rows[1], context="race.Results[2]")
    winner = _driver_name(winner_row, context="race.Results[1]")
    second = _driver_name(second_row, context="race.Results[2]")
    winner_grid = _text(winner_row.get("grid"), context="race.Results[1].grid")
    wrong_grid = "2" if winner_grid != "2" else "1"
    prefix = f"r{int(session.round):02d}"
    event = session.expected_race_name
    gold = [session.document_id]
    return [
        Claim(
            claim_id=f"{prefix}_supported_winner",
            claim=f"{winner} won the 2023 {event} race.",
            gold_label=VerdictLabel.SUPPORTED,
            gold_document_ids=gold,
            notes="Directly derived from race position 1.",
        ),
        Claim(
            claim_id=f"{prefix}_supported_second",
            claim=f"{second} finished second in the 2023 {event} race.",
            gold_label=VerdictLabel.SUPPORTED,
            gold_document_ids=gold,
            notes="Directly derived from race position 2.",
        ),
        Claim(
            claim_id=f"{prefix}_refuted_winner",
            claim=f"{second} won the 2023 {event} race.",
            gold_label=VerdictLabel.REFUTED,
            gold_document_ids=gold,
            notes="Controlled mutation: the recorded second-place driver is asserted as winner.",
        ),
        Claim(
            claim_id=f"{prefix}_refuted_grid",
            claim=f"{winner} started the 2023 {event} race from grid position {wrong_grid}.",
            gold_label=VerdictLabel.REFUTED,
            gold_document_ids=gold,
            notes=f"Controlled mutation of recorded grid position {winner_grid}.",
        ),
        Claim(
            claim_id=f"{prefix}_nee_pit_stops",
            claim=f"{winner} made exactly two pit stops in the 2023 {event} race.",
            gold_label=VerdictLabel.NOT_ENOUGH_EVIDENCE,
            notes="Pit-stop counts are intentionally absent from every pilot document.",
        ),
        Claim(
            claim_id=f"{prefix}_nee_tyre",
            claim=f"{winner} used soft tyres for the final stint of the 2023 {event} race.",
            gold_label=VerdictLabel.NOT_ENOUGH_EVIDENCE,
            notes="Tyre compounds and stints are intentionally absent from every pilot document.",
        ),
    ]


def _load_download_records(raw_directory: Path) -> dict[str, dict[str, Any]]:
    manifest_path = raw_directory / "download_manifest.json"
    payload = parse_json_bytes(manifest_path.read_bytes(), source=str(manifest_path))
    records = _list(payload.get("records"), context="download_manifest.records")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        item = _object(record, context=f"download_manifest.records[{index}]")
        source_id = _text(item.get("source_id"), context=f"records[{index}].source_id")
        if source_id in by_id:
            raise PilotDataError(f"Duplicate downloaded source_id: {source_id}")
        by_id[source_id] = item
    return by_id


def _relative_file_path(document_path: Path, manifest_path: Path) -> str:
    return Path(os.path.relpath(document_path, manifest_path.parent)).as_posix()


def build_pilot(
    plan_path: Path,
    raw_directory: Path,
    project_root: Path,
) -> None:
    """Build local corpus files, manifests, provenance, and the reviewed-size claim set."""

    plan = load_source_plan(plan_path)
    records = _load_download_records(raw_directory)
    clean_manifest_path = project_root / "data/manifests/local/f1_2023_pilot_v3_clean.jsonl"
    noisy_manifest_path = project_root / "data/manifests/local/f1_2023_pilot_v3_noisy.jsonl"
    claims_path = project_root / "data/ground_truth/f1_2023_pilot.jsonl"
    provenance_path = project_root / "data/provenance/local/f1_2023_pilot_v3.json"
    clean_directory = project_root / "data/corpora/local/f1_2023_pilot_v3/clean"
    noise_directory = project_root / "data/corpora/local/f1_2023_pilot_v3/noise"
    targets = [
        clean_manifest_path,
        noisy_manifest_path,
        provenance_path,
        clean_directory,
        noise_directory,
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing pilot outputs: " + ", ".join(existing)
        )

    prepared: list[tuple[SessionSource, dict[str, Any], str, Path]] = []
    all_claims: list[Claim] = []
    for session in plan.sessions:
        record = records.get(session.source_id)
        if record is None:
            raise PilotDataError(f"Missing downloaded record for {session.source_id}")
        if record.get("url") != session.url or record.get("raw_file") != session.raw_file:
            raise PilotDataError(f"Download manifest mismatch for {session.source_id}")
        raw_path = raw_directory / session.raw_file
        expected_hash = _text(record.get("sha256"), context=f"{session.source_id}.sha256")
        if not raw_path.is_file() or sha256_file(raw_path) != expected_hash:
            raise PilotDataError(f"Raw response hash mismatch for {session.source_id}: {raw_path}")
        payload = parse_json_bytes(raw_path.read_bytes(), source=session.url)
        race = validate_session_payload(session, payload)
        rendered = render_document(session, race, plan.source)
        directory = clean_directory if session.corpus_role == "clean" else noise_directory
        target = directory / session.document_file
        prepared.append((session, race, rendered, target))
        if session.corpus_role == "clean":
            all_claims.extend(derive_claims(session, race))

    if len(all_claims) != 18:
        raise PilotDataError(f"Expected exactly 18 pilot claims, derived {len(all_claims)}")
    claim_records = [claim.model_dump(mode="json") for claim in all_claims]
    if claims_path.exists():
        existing_claims = [Claim.model_validate(raw) for _, raw in read_jsonl(claims_path)]
        existing_records = [claim.model_dump(mode="json") for claim in existing_claims]
        if existing_records != claim_records:
            raise PilotDataError(
                f"Existing reviewed claim set differs from deterministic derivation: {claims_path}"
            )

    clean_directory.mkdir(parents=True, exist_ok=False)
    noise_directory.mkdir(parents=True, exist_ok=False)
    clean_records: list[dict[str, object]] = []
    noise_records: list[dict[str, object]] = []
    generated_hashes: dict[str, str] = {}
    for session, race, rendered, target in prepared:
        atomic_write_text(target, rendered)
        generated_hashes[target.relative_to(project_root).as_posix()] = sha256_text(rendered)
        event_date = _text(race.get("date"), context=f"{session.source_id}.date")
        title_type = {
            "race": "Race",
            "qualifying": "Qualifying",
            "sprint": "Sprint",
        }[session.session_type]
        declared_manifest = (
            clean_manifest_path if session.corpus_role == "clean" else noisy_manifest_path
        )
        document = Document(
            document_id=session.document_id,
            title=f"2023 {session.expected_race_name} — {title_type} classification",
            source=f"{plan.source.name} ({plan.source.license}); {session.url}",
            event_date=event_date,
            topic=f"2023 Formula One round {session.round} {title_type.lower()}",
            language="en",
            file_path=Path(_relative_file_path(target, declared_manifest)),
            corpus_tags=[
                session.corpus_role,
                "f1-2023-pilot",
                "jolpica",
                session.session_type,
            ],
        )
        record = document.model_dump(mode="json")
        if session.corpus_role == "clean":
            clean_records.append(record)
        else:
            noise_records.append(record)

    noisy_clean_records = []
    for record in clean_records:
        clean_path = clean_directory / Path(str(record["file_path"])).name
        noisy_record = dict(record)
        noisy_record["file_path"] = _relative_file_path(clean_path, noisy_manifest_path)
        noisy_clean_records.append(noisy_record)
    write_jsonl(clean_manifest_path, clean_records)
    write_jsonl(noisy_manifest_path, [*noisy_clean_records, *noise_records])
    if not claims_path.exists():
        write_jsonl(claims_path, claim_records)
    generated_hashes[clean_manifest_path.relative_to(project_root).as_posix()] = sha256_file(
        clean_manifest_path
    )
    generated_hashes[noisy_manifest_path.relative_to(project_root).as_posix()] = sha256_file(
        noisy_manifest_path
    )
    generated_hashes[claims_path.relative_to(project_root).as_posix()] = sha256_file(claims_path)
    write_json(
        provenance_path,
        {
            "schema_version": 1,
            "transformer_version": TRANSFORMER_VERSION,
            "classification_position_limit": MAX_CLASSIFICATION_POSITIONS,
            "source_plan": plan_path.relative_to(project_root).as_posix(),
            "source_plan_sha256": sha256_file(plan_path),
            "raw_download_manifest_sha256": sha256_file(raw_directory / "download_manifest.json"),
            "license": plan.source.license,
            "license_url": plan.source.license_url,
            "terms_url": plan.source.terms_url,
            "generated_sha256": dict(sorted(generated_hashes.items())),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "build", "all"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> int:
    """Run one explicit, non-overwriting data-preparation action."""

    args = _parse_args()
    project_root = args.project_root.resolve()
    plan_path = project_root / "data/sources/f1_2023_pilot_jolpica.yaml"
    raw_directory = project_root / "data/raw/local/f1_2023_pilot"
    if args.action in {"download", "all"}:
        asyncio.run(download_sources(plan_path, raw_directory))
    if args.action in {"build", "all"}:
        build_pilot(plan_path, raw_directory, project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
