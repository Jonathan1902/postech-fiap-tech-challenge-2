"""Treino de cada candidato e registro do run no MLflow."""

from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from shopper_intent.models.evaluator import Evaluator

MODEL_ARTIFACT_NAME = "model"


@dataclass(frozen=True)
class TrainingData:
    """Conjuntos de treino e teste já separados em features e alvo."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class TrainResult:
    """Resultado do treino de um candidato."""

    name: str
    run_id: str
    metrics: dict[str, float]
    pipeline: Pipeline


class Trainer:
    """Treina um Pipeline e loga parâmetros, métricas e modelo no MLflow."""

    def __init__(self, tracking_uri: str, experiment_name: str, evaluator: Evaluator) -> None:
        """Guarda a URI do servidor MLflow, o nome do experimento e o avaliador."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.evaluator = evaluator

    def train_and_log(
        self,
        name: str,
        pipeline: Pipeline,
        params: dict[str, Any],
        data: TrainingData,
    ) -> TrainResult:
        """Treina o candidato, avalia no teste e loga um run no MLflow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model": name, **params})
            pipeline.fit(data.x_train, data.y_train)
            y_proba = pipeline.predict_proba(data.x_test)[:, 1]
            metrics = self.evaluator.evaluate(data.y_test, y_proba)
            mlflow.log_metrics(metrics)
            self._log_model(pipeline, data.x_train)
        return TrainResult(name, run.info.run_id, metrics, pipeline)

    def _log_model(self, pipeline: Pipeline, x_sample: pd.DataFrame) -> None:
        """Loga o Pipeline com assinatura e exemplo de entrada."""
        example = x_sample.head(5)
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=infer_signature(example, pipeline.predict_proba(example)[:, 1]),
            input_example=example,
            serialization_format="cloudpickle",
        )
