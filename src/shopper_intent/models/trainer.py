"""Treino de cada candidato e registro do run no MLflow."""

from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from shopper_intent.models.evaluator import Evaluator
from shopper_intent.models.validation import CrossValidation

MODEL_ARTIFACT_NAME = "model"
CONFUSION_MATRIX_ARTIFACT = "confusion_matrix.json"


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
    cv_score: float
    metrics: dict[str, float]
    pipeline: Pipeline


@dataclass(frozen=True)
class Trainer:
    """Treina um Pipeline e loga parâmetros, métricas, tags e modelo no MLflow."""

    tracking_uri: str
    experiment_name: str
    evaluator: Evaluator
    cross_validation: CrossValidation
    tags: dict[str, str] = field(default_factory=dict)

    def train_and_log(
        self, name: str, pipeline: Pipeline, params: dict[str, Any], data: TrainingData
    ) -> TrainResult:
        """Valida no treino (CV), treina, avalia no teste e loga um run no MLflow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(self.tags)
            mlflow.log_params({**self._run_params(name), **params})
            cv_score = self.cross_validation.score(pipeline, data.x_train, data.y_train)
            mlflow.log_metric(f"cv_{self.cross_validation.scoring}", cv_score)
            pipeline.fit(data.x_train, data.y_train)
            y_proba = pipeline.predict_proba(data.x_test)[:, 1]
            metrics = self.evaluator.evaluate(data.y_test, y_proba)
            mlflow.log_metrics(metrics)
            confusion = self.evaluator.confusion(data.y_test, y_proba)
            mlflow.log_dict(confusion, CONFUSION_MATRIX_ARTIFACT)
            self._log_model(pipeline, data.x_train)
        return TrainResult(name, run.info.run_id, cv_score, metrics, pipeline)

    def _run_params(self, name: str) -> dict[str, Any]:
        """Parâmetros comuns a todos os runs: modelo, seed, folds e limiar."""
        return {
            "model": name,
            "seed": self.cross_validation.seed,
            "cv_folds": self.cross_validation.folds,
            "threshold": self.evaluator.threshold,
        }

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
