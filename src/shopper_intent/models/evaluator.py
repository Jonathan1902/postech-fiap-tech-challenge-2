"""Cálculo das métricas de classificação."""

import numpy as np
import pandas as pd


class Evaluator:
    """Calcula as métricas de classificação a partir de probabilidades previstas."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Guarda o limiar usado para converter probabilidade em classe."""
        self.threshold = threshold

    def evaluate(self, y_true: pd.Series, y_proba: np.ndarray) -> dict[str, float]:
        """Devolve PR-AUC, ROC-AUC, F1, precision e recall como dicionário."""
        raise NotImplementedError
