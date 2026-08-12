"""Re-evaluate a persisted benchmark run without invoking retrieval or a model."""

from pathlib import Path

from pydantic import ValidationError

from rag_claim_verification.errors import ConfigurationError
from rag_claim_verification.evaluation.reporting import write_evaluation_artifacts
from rag_claim_verification.models.prediction import Prediction
from rag_claim_verification.models.run import CaseManifestRecord, RunMetadata
from rag_claim_verification.utils.files import FileFormatError, read_jsonl, read_yaml
from rag_claim_verification.utils.hashing import sha256_file


def evaluate_run(run_directory: Path) -> dict[str, object]:
    """Reload predictions.jsonl and regenerate all derived evaluation artifacts."""

    resolved = run_directory.resolve()
    predictions_path = resolved / "predictions.jsonl"
    case_manifest_path = resolved / "case_manifest.jsonl"
    metadata_path = resolved / "metadata.json"
    if not predictions_path.is_file():
        raise ConfigurationError(f"Run has no predictions.jsonl: {resolved}")
    if not case_manifest_path.is_file():
        raise ConfigurationError(f"Run has no case_manifest.jsonl: {resolved}")
    if not metadata_path.is_file():
        raise ConfigurationError(f"Run has no metadata.json: {resolved}")
    try:
        metadata = RunMetadata.model_validate(read_yaml(metadata_path))
    except (FileFormatError, ValidationError) as exc:
        raise ConfigurationError(f"Invalid run metadata {metadata_path}:\n{exc}") from exc
    if metadata.case_manifest_hash != sha256_file(case_manifest_path):
        raise ConfigurationError(f"case_manifest.jsonl hash does not match {metadata_path}")
    if metadata.predictions_hash is None:
        raise ConfigurationError(f"Run metadata has no final predictions hash: {metadata_path}")
    if metadata.predictions_hash != sha256_file(predictions_path):
        raise ConfigurationError(f"predictions.jsonl hash does not match {metadata_path}")
    resolved_config_path = resolved / "resolved_config.yaml"
    if not resolved_config_path.is_file() or metadata.resolved_config_hash != sha256_file(
        resolved_config_path
    ):
        raise ConfigurationError(f"resolved_config.yaml hash does not match {metadata_path}")
    claims_snapshot_path = resolved / "inputs/claims.jsonl"
    if not claims_snapshot_path.is_file() or metadata.claims_snapshot_hash != sha256_file(
        claims_snapshot_path
    ):
        raise ConfigurationError(f"claims snapshot hash does not match {metadata_path}")
    try:
        records = read_jsonl(predictions_path)
    except FileFormatError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not records:
        raise ConfigurationError(f"Run contains no predictions: {resolved}")
    try:
        case_records = read_jsonl(case_manifest_path)
    except FileFormatError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not case_records:
        raise ConfigurationError(f"Run contains no planned cases: {resolved}")
    cases: list[CaseManifestRecord] = []
    for line_number, raw in case_records:
        try:
            cases.append(CaseManifestRecord.model_validate(raw))
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid case manifest record at {case_manifest_path}:{line_number}:\n{exc}"
            ) from exc
    predictions: list[Prediction] = []
    for line_number, raw in records:
        try:
            predictions.append(Prediction.model_validate(raw))
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid prediction at {predictions_path}:{line_number}:\n{exc}"
            ) from exc
    try:
        return write_evaluation_artifacts(resolved, predictions, cases)
    except ValueError as exc:
        raise ConfigurationError(f"Cannot evaluate incomplete or inconsistent run: {exc}") from exc
