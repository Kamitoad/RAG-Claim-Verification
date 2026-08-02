"""Evaluation artifact generation from persisted predictions."""

import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_claim_verification.evaluation.classification_metrics import (
    LABELS,
    NO_PREDICTION,
    compute_classification_metrics,
)
from rag_claim_verification.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_claim_verification.models.claim import VerdictLabel
from rag_claim_verification.models.prediction import Prediction
from rag_claim_verification.utils.files import atomic_write_text, write_json, write_jsonl


def evaluate_predictions(predictions: list[Prediction]) -> dict[str, Any]:
    """Compute classification and conditionally available retrieval metrics by condition."""

    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.condition].append(prediction)
    metrics: dict[str, Any] = {"conditions": {}}
    for condition, items in sorted(grouped.items()):
        if any(item.gold_label is None for item in items):
            raise ValueError(f"Condition {condition} contains predictions without gold labels")
        gold = [item.gold_label for item in items if item.gold_label is not None]
        predicted = [item.predicted_label for item in items]
        classification = compute_classification_metrics(
            gold,
            predicted,
            parse_error_count=sum(item.parse_error is not None for item in items),
            pipeline_error_count=sum(item.error is not None for item in items),
        )
        retrieval = _retrieval_metrics_or_limitation(items)
        metrics["conditions"][condition] = {
            "classification": classification,
            "retrieval": retrieval,
        }
    return metrics


def _retrieval_metrics_or_limitation(items: list[Prediction]) -> dict[str, Any] | None:
    if all(item.verification_mode == "baseline" for item in items):
        return None
    eligible = [item for item in items if item.gold_document_ids]
    if not eligible:
        return {
            "available": False,
            "reason": "No predictions contain gold_document_ids.",
        }
    if any(not item.retrieval_supports_document_ids for item in eligible):
        return {
            "available": False,
            "reason": (
                "At least one eligible retrieval could not be mapped to concrete document IDs; "
                "metrics were not simulated."
            ),
        }
    gold_ids = [set(item.gold_document_ids) for item in eligible]
    retrieved_ids = [
        [evidence.document_id for evidence in item.evidence if evidence.document_id is not None]
        for item in eligible
    ]
    return {"available": True, **compute_retrieval_metrics(gold_ids, retrieved_ids)}


def write_evaluation_artifacts(
    run_directory: Path, predictions: list[Prediction]
) -> dict[str, Any]:
    """Write metrics, CSV views, failures, and a non-speculative Markdown summary."""

    metrics = evaluate_predictions(predictions)
    write_json(run_directory / "metrics.json", metrics)
    _write_metrics_csv(run_directory / "metrics.csv", metrics)
    _write_confusion_csv(run_directory / "confusion_matrix.csv", metrics)
    failures = build_failure_records(predictions)
    write_jsonl(run_directory / "failures.jsonl", failures)
    atomic_write_text(run_directory / "summary.md", build_summary(metrics, failures))
    return metrics


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    stream = io.StringIO(newline="")
    fieldnames = [
        "condition",
        "accuracy",
        "macro_f1",
        "sample_count",
        "valid_prediction_count",
        "parse_error_count",
        "parse_error_rate",
        "pipeline_error_count",
        "class",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for condition, payload in metrics["conditions"].items():
        classification = payload["classification"]
        for label, per_class in classification["per_class"].items():
            writer.writerow(
                {
                    "condition": condition,
                    "accuracy": classification["accuracy"],
                    "macro_f1": classification["macro_f1"],
                    "sample_count": classification["sample_count"],
                    "valid_prediction_count": classification["valid_prediction_count"],
                    "parse_error_count": classification["parse_errors"]["count"],
                    "parse_error_rate": classification["parse_errors"]["rate"],
                    "pipeline_error_count": classification["pipeline_errors"]["count"],
                    "class": label,
                    **per_class,
                }
            )
    atomic_write_text(path, stream.getvalue())


def _write_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    stream = io.StringIO(newline="")
    columns = [label.value for label in LABELS] + [NO_PREDICTION]
    writer = csv.DictWriter(stream, fieldnames=["condition", "gold_label", *columns])
    writer.writeheader()
    for condition, payload in metrics["conditions"].items():
        matrix = payload["classification"]["confusion_matrix"]
        for gold_label, row in matrix.items():
            writer.writerow({"condition": condition, "gold_label": gold_label, **row})
    atomic_write_text(path, stream.getvalue())


def build_failure_records(predictions: list[Prediction]) -> list[dict[str, Any]]:
    """Classify observable failure patterns without claiming an unobserved causal mechanism."""

    by_claim: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_claim[prediction.claim_id].append(prediction)
    failures: list[dict[str, Any]] = []
    for prediction in predictions:
        is_incorrect = (
            prediction.gold_label is not None
            and prediction.predicted_label != prediction.gold_label
        )
        if not is_incorrect and prediction.parse_error is None and prediction.error is None:
            continue
        categories: list[str] = []
        if prediction.parse_error is not None:
            categories.append("invalid_model_output")
        if prediction.error is not None:
            categories.append("pipeline_error")
        retrieved = {
            item.document_id for item in prediction.evidence if item.document_id is not None
        }
        gold = set(prediction.gold_document_ids)
        if is_incorrect and gold:
            if retrieved & gold:
                categories.append("relevant_evidence_retrieved_but_verdict_incorrect")
            elif prediction.retrieval_supports_document_ids:
                categories.append("relevant_evidence_not_retrieved")
        if (
            is_incorrect
            and prediction.verification_mode == "rag"
            and prediction.predicted_label != VerdictLabel.NOT_ENOUGH_EVIDENCE
            and not prediction.evidence
        ):
            categories.append("verdict_without_retrieved_evidence")
        peers = by_claim[prediction.claim_id]
        if (
            is_incorrect
            and "noisy" in prediction.condition.casefold()
            and gold
            and not retrieved & gold
            and any(
                peer.condition != prediction.condition
                and any(
                    evidence.document_id in gold
                    for evidence in peer.evidence
                    if evidence.document_id is not None
                )
                for peer in peers
            )
        ):
            categories.append("relevant_evidence_present_in_other_condition_only")
        failures.append(
            {
                "claim_id": prediction.claim_id,
                "condition": prediction.condition,
                "gold_label": (
                    prediction.gold_label.value if prediction.gold_label is not None else None
                ),
                "predicted_label": (
                    prediction.predicted_label.value
                    if prediction.predicted_label is not None
                    else None
                ),
                "categories": categories or ["incorrect_verdict"],
                "gold_document_ids": prediction.gold_document_ids,
                "retrieved_document_ids": [
                    item.document_id for item in prediction.evidence if item.document_id is not None
                ],
                "cited_document_ids": prediction.cited_document_ids,
                "reason": prediction.reason,
                "parse_error": prediction.parse_error,
                "error": prediction.error,
            }
        )
    return failures


def build_summary(metrics: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    """Create a compact descriptive summary with no causal or scientific extrapolation."""

    lines = [
        "# Benchmark summary",
        "",
        "This file describes the recorded predictions only. It does not establish statistical "
        "significance or generalize beyond the evaluated claims.",
        "",
        "## Conditions",
        "",
        "| Condition | N | Accuracy | Macro-F1 | Parse errors | Pipeline errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, payload in metrics["conditions"].items():
        classification = payload["classification"]
        lines.append(
            f"| {condition} | {classification['sample_count']} | "
            f"{classification['accuracy']:.4f} | {classification['macro_f1']:.4f} | "
            f"{classification['parse_errors']['count']} | "
            f"{classification['pipeline_errors']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Recorded failures",
            "",
            f"`failures.jsonl` contains {len(failures)} incorrect or failed prediction records.",
            "See `metrics.json` for class-specific and retrieval metrics, including explicit "
            "availability reasons when retrieval metrics could not be computed.",
            "",
        ]
    )
    return "\n".join(lines)
