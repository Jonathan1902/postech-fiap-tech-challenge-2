"""Testes do Evaluator em um caso conhecido."""

import numpy as np
import pandas as pd
import pytest

from shopper_intent.models.evaluator import Evaluator


def test_evaluate_known_case() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = Evaluator(threshold=0.5).evaluate(y_true, y_proba)

    # Predições: [0, 1, 0, 1] -> TP=1, FP=1, FN=1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(0.8333333)


def test_evaluate_without_positive_predictions_does_not_raise() -> None:
    metrics = Evaluator(threshold=0.99).evaluate(pd.Series([0, 1]), np.array([0.1, 0.2]))

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_confusion_counts_match_threshold() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])

    confusion = Evaluator(threshold=0.5).confusion(y_true, y_proba)

    assert confusion == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
