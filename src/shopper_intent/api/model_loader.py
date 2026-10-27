"""Carregamento do modelo promovido (alias) no MLflow Model Registry."""

from dataclasses import dataclass
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from shopper_intent.config import Settings

DEFAULT_THRESHOLD = 0.5


class ProbabilisticClassifier(Protocol):
    """Qualquer estimador que devolva probabilidades por classe."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Devolve uma matriz (n_amostras, n_classes) de probabilidades."""
        ...


@dataclass(frozen=True)
class LoadedModel:
    """Modelo pronto para inferência, com a versão e o limiar usados no treino."""

    classifier: ProbabilisticClassifier
    version: str
    threshold: float

    def purchase_probability(self, features: pd.DataFrame) -> float:
        """Devolve a probabilidade da classe positiva (compra) para uma sessão."""
        return float(self.classifier.predict_proba(features)[0, 1])


def load_champion(settings: Settings) -> LoadedModel:
    """Carrega a versão apontada pelo alias e o limiar logado no run que a gerou."""
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_version = client.get_model_version_by_alias(
        settings.registered_model_name, settings.model_alias
    )
    params = client.get_run(model_version.run_id).data.params
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Carrega pelo número da versão (e não pelo alias) para que versão e modelo sejam a mesma.
    classifier = mlflow.sklearn.load_model(
        f"models:/{settings.registered_model_name}/{model_version.version}"
    )
    threshold = float(params.get("threshold", DEFAULT_THRESHOLD))
    return LoadedModel(classifier, str(model_version.version), threshold)
