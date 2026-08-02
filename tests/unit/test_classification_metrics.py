"""Classification metric tests with hand-computed expectations."""

import pytest

from rag_claim_verification.evaluation.classification_metrics import (
    accuracy_score,
    confusion_matrix,
    macro_f1_score,
)
from rag_claim_verification.models.claim import VerdictLabel


def test_accuracy_counts_missing_prediction_as_incorrect() -> None:
    gold = [VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_EVIDENCE]
    predicted = [VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, None]

    assert accuracy_score(gold, predicted) == pytest.approx(1 / 3)


def test_macro_f1_is_unweighted_across_all_classes() -> None:
    gold = [VerdictLabel.SUPPORTED, VerdictLabel.REFUTED, VerdictLabel.NOT_ENOUGH_EVIDENCE]
    predicted = [VerdictLabel.SUPPORTED, VerdictLabel.SUPPORTED, None]

    # SUPPORTED F1 = 2/3; the other two F1 values are zero.
    assert macro_f1_score(gold, predicted) == pytest.approx(2 / 9)


def test_confusion_matrix_includes_no_prediction_column() -> None:
    gold = [VerdictLabel.SUPPORTED, VerdictLabel.REFUTED]
    predicted = [VerdictLabel.REFUTED, None]

    matrix = confusion_matrix(gold, predicted)

    assert matrix["SUPPORTED"]["REFUTED"] == 1
    assert matrix["REFUTED"]["NO_PREDICTION"] == 1
