"""Evaluation artifact generation from persisted raw case results."""

import csv
import io
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag_claim_verification.evaluation.classification_metrics import (
    LABELS,
    NO_PREDICTION,
    compute_classification_metrics,
)
from rag_claim_verification.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_claim_verification.models.claim import VerdictLabel
from rag_claim_verification.models.prediction import ParseStatus, Prediction
from rag_claim_verification.models.run import CaseManifestRecord
from rag_claim_verification.utils.files import atomic_write_text, write_json, write_jsonl


def validate_case_completeness(
    cases: list[CaseManifestRecord], predictions: list[Prediction]
) -> dict[str, Any]:
    """Require exactly one raw result for every planned claim-condition case."""

    if not cases:
        raise ValueError("case manifest contains no planned cases")
    expected_ids = [case.case_id for case in cases]
    actual_ids = [prediction.case_id for prediction in predictions]
    duplicate_expected = sorted(
        case_id for case_id, count in Counter(expected_ids).items() if count > 1
    )
    duplicate_actual = sorted(
        case_id for case_id, count in Counter(actual_ids).items() if count > 1
    )
    if duplicate_expected:
        raise ValueError(
            "case manifest contains duplicate case IDs: " + ", ".join(duplicate_expected)
        )
    if duplicate_actual:
        raise ValueError("predictions contain duplicate case IDs: " + ", ".join(duplicate_actual))
    missing = sorted(set(expected_ids) - set(actual_ids))
    unexpected = sorted(set(actual_ids) - set(expected_ids))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError("prediction set does not match case manifest: " + "; ".join(details))
    if actual_ids != expected_ids:
        raise ValueError("prediction order does not match case manifest sequence")

    expected_by_id = {case.case_id: case for case in cases}
    for prediction in predictions:
        expected = expected_by_id[prediction.case_id]
        gold_label = prediction.gold_label.value if prediction.gold_label is not None else None
        if (
            prediction.claim_id != expected.claim_id
            or prediction.claim != expected.claim
            or prediction.condition != expected.condition
            or prediction.verification_mode != expected.verification_mode
            or gold_label != expected.gold_label.value
            or prediction.gold_document_ids != expected.gold_document_ids
        ):
            raise ValueError(f"prediction {prediction.case_id!r} does not match its planned case")
    return {
        "planned_case_count": len(cases),
        "recorded_case_count": len(predictions),
        "missing_case_count": 0,
        "unexpected_case_count": 0,
        "complete": True,
    }


def evaluate_predictions(
    predictions: list[Prediction], cases: list[CaseManifestRecord]
) -> dict[str, Any]:
    """Compute deterministic metrics while preserving retrieval/verdict separation."""

    completeness = validate_case_completeness(cases, predictions)
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.condition].append(prediction)
    condition_order = list(dict.fromkeys(case.condition for case in cases))
    metrics: dict[str, Any] = {
        "schema_version": "2.0",
        "completeness": completeness,
        "conditions": {},
    }
    for condition in condition_order:
        items = grouped[condition]
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
        metrics["conditions"][condition] = {
            "classification": classification,
            "retrieval": _retrieval_metrics_or_limitation(items),
            "grounding": _grounding_metrics(items),
            "structured_output": _structured_output_metrics(items),
            "technical_errors": _technical_error_metrics(items),
            "latency": _latency_metrics(items),
        }
    return metrics


def _retrieval_metrics_or_limitation(items: list[Prediction]) -> dict[str, Any] | None:
    if all(item.verification_mode == "baseline" for item in items):
        return None
    status_counts = dict(sorted(Counter(item.retrieval_status.value for item in items).items()))
    eligible = [item for item in items if item.gold_document_ids]
    common: dict[str, Any] = {
        "case_count": len(items),
        "annotation_eligible_count": len(eligible),
        "annotation_ineligible_count": len(items) - len(eligible),
        "status_counts": status_counts,
    }
    if not eligible:
        return {
            **common,
            "available": False,
            "measurable_count": 0,
            "reason": "No predictions contain gold_document_ids.",
        }
    unmappable = [item.case_id for item in eligible if not item.retrieval_supports_document_ids]
    if unmappable:
        return {
            **common,
            "available": False,
            "measurable_count": len(eligible) - len(unmappable),
            "unmappable_case_ids": unmappable,
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
    return {
        **common,
        "available": True,
        "measurable_count": len(eligible),
        **compute_retrieval_metrics(gold_ids, retrieved_ids),
    }


def _grounding_metrics(items: list[Prediction]) -> dict[str, Any] | None:
    if all(item.verification_mode == "baseline" for item in items):
        return None
    valid = [item for item in items if item.predicted_label is not None]
    with_citations = [item for item in valid if item.cited_document_ids]
    decisive = [
        item
        for item in valid
        if item.predicted_label in {VerdictLabel.SUPPORTED, VerdictLabel.REFUTED}
    ]
    decisive_with_citations = [item for item in decisive if item.cited_document_ids]
    citation_valid = [
        item
        for item in valid
        if set(item.cited_document_ids)
        <= {evidence.document_id for evidence in item.evidence if evidence.document_id is not None}
    ]
    gold_eligible = [item for item in valid if item.gold_document_ids]
    gold_hits = [
        item for item in gold_eligible if set(item.cited_document_ids) & set(item.gold_document_ids)
    ]
    return {
        "valid_prediction_count": len(valid),
        "citation_present_count": len(with_citations),
        "citation_presence_rate": _rate(len(with_citations), len(valid)),
        "citation_allowlist_valid_count": len(citation_valid),
        "citation_allowlist_valid_rate": _rate(len(citation_valid), len(valid)),
        "decisive_prediction_count": len(decisive),
        "decisive_with_citation_count": len(decisive_with_citations),
        "decisive_citation_rate": _rate(len(decisive_with_citations), len(decisive)),
        "gold_citation_eligible_count": len(gold_eligible),
        "gold_citation_hit_count": len(gold_hits),
        "gold_citation_hit_rate": _rate(len(gold_hits), len(gold_eligible)),
        "verdict_without_evidence_count": sum(
            item.predicted_label not in {None, VerdictLabel.NOT_ENOUGH_EVIDENCE}
            and not item.evidence
            for item in items
        ),
    }


def _structured_output_metrics(items: list[Prediction]) -> dict[str, Any]:
    statuses = Counter(item.parse_status.value for item in items)
    attempted_repairs = sum(item.initial_parse_error is not None for item in items)
    successful_repairs = statuses[ParseStatus.VALID_AFTER_REPAIR.value]
    final_valid = statuses[ParseStatus.VALID_FIRST_PASS.value] + successful_repairs
    return {
        "case_count": len(items),
        "status_counts": dict(sorted(statuses.items())),
        "first_pass_valid_count": statuses[ParseStatus.VALID_FIRST_PASS.value],
        "first_pass_valid_rate": _rate(statuses[ParseStatus.VALID_FIRST_PASS.value], len(items)),
        "repair_attempt_count": attempted_repairs,
        "repair_success_count": successful_repairs,
        "repair_success_rate": _rate(successful_repairs, attempted_repairs),
        "final_valid_count": final_valid,
        "final_valid_rate": _rate(final_valid, len(items)),
        "final_invalid_count": statuses[ParseStatus.INVALID_AFTER_REPAIR.value],
    }


def _technical_error_metrics(items: list[Prediction]) -> dict[str, Any]:
    failures = [item for item in items if item.error is not None]
    return {
        "count": len(failures),
        "rate": _rate(len(failures), len(items)),
        "case_status_counts": dict(
            sorted(Counter(item.case_status.value for item in items).items())
        ),
        "by_stage": dict(
            sorted(Counter(item.error_stage for item in failures if item.error_stage).items())
        ),
        "by_type": dict(
            sorted(Counter(item.error_type for item in failures if item.error_type).items())
        ),
    }


def _latency_metrics(items: list[Prediction]) -> dict[str, Any]:
    return {
        "total_ms": _distribution([item.latency_ms for item in items]),
        "retrieval_ms": _distribution(
            [item.retrieval_latency_ms for item in items if item.verification_mode == "rag"]
        ),
        "generation_ms": _distribution(
            [
                item.generation_latency_ms
                for item in items
                if item.model_calls or item.error_stage == "model"
            ]
        ),
        "repair_ms": _distribution(
            [item.repair_latency_ms for item in items if item.initial_parse_error is not None]
        ),
    }


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95_nearest_rank": None,
            "max": None,
        }
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p95_nearest_rank": ordered[p95_index],
        "max": ordered[-1],
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_evaluation_artifacts(
    run_directory: Path,
    predictions: list[Prediction],
    cases: list[CaseManifestRecord],
) -> dict[str, Any]:
    """Write derived metrics separately from immutable raw case observations."""

    metrics = evaluate_predictions(predictions, cases)
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
        "technical_error_count",
        "technical_error_rate",
        "first_pass_valid_rate",
        "repair_success_rate",
        "median_latency_ms",
        "p95_latency_ms",
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
        structured = payload["structured_output"]
        technical = payload["technical_errors"]
        latency = payload["latency"]["total_ms"]
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
                    "technical_error_count": technical["count"],
                    "technical_error_rate": technical["rate"],
                    "first_pass_valid_rate": structured["first_pass_valid_rate"],
                    "repair_success_rate": structured["repair_success_rate"],
                    "median_latency_ms": latency["median"],
                    "p95_latency_ms": latency["p95_nearest_rank"],
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
    """Classify observable failure patterns without claiming an unobserved cause."""

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
            categories.append("technical_error")
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
                "schema_version": "2.0",
                "case_id": prediction.case_id,
                "claim_id": prediction.claim_id,
                "condition": prediction.condition,
                "case_status": prediction.case_status.value,
                "retrieval_status": prediction.retrieval_status.value,
                "parse_status": prediction.parse_status.value,
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
                "initial_parse_error": prediction.initial_parse_error,
                "parse_error": prediction.parse_error,
                "error": prediction.error,
                "error_stage": prediction.error_stage,
                "error_type": prediction.error_type,
            }
        )
    return failures


def build_summary(metrics: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    """Create a compact descriptive summary with no causal extrapolation."""

    completeness = metrics["completeness"]
    lines = [
        "# Benchmark summary",
        "",
        "This file describes the recorded predictions only. It does not establish statistical "
        "significance or generalize beyond the evaluated claims.",
        "",
        "## Completeness",
        "",
        f"Recorded {completeness['recorded_case_count']} of "
        f"{completeness['planned_case_count']} planned cases.",
        "",
        "## Conditions",
        "",
        "| Condition | N | Accuracy | Macro-F1 | First-pass valid | Repairs | Technical errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, payload in metrics["conditions"].items():
        classification = payload["classification"]
        structured = payload["structured_output"]
        technical = payload["technical_errors"]
        lines.append(
            f"| {condition} | {classification['sample_count']} | "
            f"{classification['accuracy']:.4f} | {classification['macro_f1']:.4f} | "
            f"{structured['first_pass_valid_count']} | "
            f"{structured['repair_attempt_count']} | {technical['count']} |"
        )
    lines.extend(
        [
            "",
            "## Recorded failures",
            "",
            f"`failures.jsonl` contains {len(failures)} incorrect or failed case records.",
            "See `metrics.json` for classification, retrieval, grounding, structured-output, "
            "technical-error, and latency measurements.",
            "",
        ]
    )
    return "\n".join(lines)
