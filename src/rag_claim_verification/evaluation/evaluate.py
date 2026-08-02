"""Re-evaluate a persisted benchmark run without invoking retrieval or a model."""

from pathlib import Path

from pydantic import ValidationError

from rag_claim_verification.errors import ConfigurationError
from rag_claim_verification.evaluation.reporting import write_evaluation_artifacts
from rag_claim_verification.models.prediction import Prediction
from rag_claim_verification.utils.files import FileFormatError, read_jsonl


def evaluate_run(run_directory: Path) -> dict[str, object]:
    """Reload predictions.jsonl and regenerate all derived evaluation artifacts."""

    resolved = run_directory.resolve()
    predictions_path = resolved / "predictions.jsonl"
    if not predictions_path.is_file():
        raise ConfigurationError(f"Run has no predictions.jsonl: {resolved}")
    try:
        records = read_jsonl(predictions_path)
    except FileFormatError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not records:
        raise ConfigurationError(f"Run contains no predictions: {resolved}")
    predictions: list[Prediction] = []
    for line_number, raw in records:
        try:
            predictions.append(Prediction.model_validate(raw))
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid prediction at {predictions_path}:{line_number}:\n{exc}"
            ) from exc
    return write_evaluation_artifacts(resolved, predictions)
