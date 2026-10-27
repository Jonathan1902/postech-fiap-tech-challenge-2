"""Treino de cada candidato e registro do run no MLflow."""

import pandas as pd
from sklearn.pipeline import Pipeline


class Trainer:
    """Treina um Pipeline e loga parâmetros, métricas e modelo no MLflow."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Guarda a URI do servidor MLflow e o nome do experimento."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def train_and_log(
        self,
        name: str,
        pipeline: Pipeline,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """Treina o candidato, avalia no teste, loga o run e devolve a métrica primária."""
        raise NotImplementedError
