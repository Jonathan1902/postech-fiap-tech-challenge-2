"""Cálculo das métricas de classificação."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """Calcula as métricas de classificação a partir de probabilidades previstas."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Guarda o limiar usado para converter probabilidade em classe."""
        self.threshold = threshold

    def evaluate(self, y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
        """Devolve PR-AUC, ROC-AUC, F1, precision e recall como dicionário."""
        y_pred = (y_proba >= self.threshold).astype(int)
        return {
            "average_precision": float(average_precision_score(y_true, y_proba)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
