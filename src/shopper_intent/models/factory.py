"""Catálogo de estimadores candidatos, cada um dentro de um Pipeline sklearn."""

from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


class ModelFactory:
    """Cria os estimadores candidatos a partir da configuração de params.yaml."""

    def __init__(self, candidates: dict[str, dict[str, Any]], seed: int) -> None:
        """Guarda a configuração dos candidatos e a seed."""
        unknown = set(candidates) - set(ESTIMATORS)
        if unknown:
            raise ValueError(f"Candidatos desconhecidos em params.yaml: {sorted(unknown)}")
        self.candidates = candidates
        self.seed = seed

    def names(self) -> list[str]:
        """Lista os nomes dos candidatos configurados."""
        return list(self.candidates)

    def create(self, name: str, preprocessor: ColumnTransformer) -> Pipeline:
        """Monta o Pipeline (pré-processamento + estimador) do candidato informado."""
        estimator = ESTIMATORS[name](**self.candidates[name], random_state=self.seed)
        return Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
