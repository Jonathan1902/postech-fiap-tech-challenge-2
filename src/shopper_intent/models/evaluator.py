"""Cálculo das métricas de classificação."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class Evaluator:
    """Calcula as métricas de classificação a partir de probabilidades previstas."""

    threshold: float = 0.5

    def predict_labels(self, y_proba: np.ndarray) -> np.ndarray:
        """Converte probabilidades em classes (0/1) usando o limiar configurado."""
        return (y_proba >= self.threshold).astype(int)

    def confusion(self, y_true: pd.Series, y_proba: np.ndarray) -> dict[str, int]:
        """Devolve a matriz de confusão como contagens nomeadas (tn, fp, fn, tp)."""
        matrix = confusion_matrix(y_true, self.predict_labels(y_proba), labels=[0, 1])
        tn, fp, fn, tp = (int(count) for count in matrix.ravel())
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    def evaluate(self, y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
        """Devolve PR-AUC, ROC-AUC, F1, precision e recall como dicionário."""
        y_pred = self.predict_labels(y_proba)
        return {
            "average_precision": float(average_precision_score(y_true, y_proba)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
