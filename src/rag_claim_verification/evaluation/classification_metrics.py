"""Dependency-free three-way classification metrics."""

from collections.abc import Sequence
from typing import Any

from rag_claim_verification.models.claim import VerdictLabel

LABELS = tuple(VerdictLabel)
NO_PREDICTION = "NO_PREDICTION"


def _validate_lengths(
    gold: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel | None]
) -> None:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted sequences must have equal length")
    if not gold:
        raise ValueError("at least one prediction is required")


def accuracy_score(gold: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel | None]) -> float:
    """Compute accuracy while treating missing predictions as incorrect."""

    _validate_lengths(gold, predicted)
    return sum(expected == actual for expected, actual in zip(gold, predicted, strict=True)) / len(
        gold
    )


def confusion_matrix(
    gold: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel | None]
) -> dict[str, dict[str, int]]:
    """Return gold-label rows and prediction columns, including missing outputs."""

    _validate_lengths(gold, predicted)
    columns = [label.value for label in LABELS] + [NO_PREDICTION]
    matrix = {label.value: {column: 0 for column in columns} for label in LABELS}
    for expected, actual in zip(gold, predicted, strict=True):
        column = actual.value if actual is not None else NO_PREDICTION
        matrix[expected.value][column] += 1
    return matrix


def per_class_metrics(
    gold: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel | None]
) -> dict[str, dict[str, float | int]]:
    """Compute one-vs-rest precision, recall, F1, and support for every label."""

    _validate_lengths(gold, predicted)
    result: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        true_positive = sum(
            expected == label and actual == label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        false_positive = sum(
            expected != label and actual == label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            expected == label and actual != label
            for expected, actual in zip(gold, predicted, strict=True)
        )
        support = sum(expected == label for expected in gold)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label.value] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def macro_f1_score(gold: Sequence[VerdictLabel], predicted: Sequence[VerdictLabel | None]) -> float:
    """Average the three class-specific F1 scores without support weighting."""

    metrics = per_class_metrics(gold, predicted)
    return sum(float(metrics[label.value]["f1"]) for label in LABELS) / len(LABELS)


def compute_classification_metrics(
    gold: Sequence[VerdictLabel],
    predicted: Sequence[VerdictLabel | None],
    *,
    parse_error_count: int = 0,
    pipeline_error_count: int = 0,
) -> dict[str, Any]:
    """Build the complete classification metric payload for one condition."""

    _validate_lengths(gold, predicted)
    total = len(gold)
    return {
        "sample_count": total,
        "valid_prediction_count": sum(item is not None for item in predicted),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": macro_f1_score(gold, predicted),
        "per_class": per_class_metrics(gold, predicted),
        "confusion_matrix": confusion_matrix(gold, predicted),
        "parse_errors": {
            "count": parse_error_count,
            "rate": parse_error_count / total,
        },
        "pipeline_errors": {
            "count": pipeline_error_count,
            "rate": pipeline_error_count / total,
        },
    }
