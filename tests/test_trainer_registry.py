"""Testes do Trainer e do ModelRegistry contra um MLflow em SQLite temporário."""

import mlflow
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from shopper_intent.models.evaluator import Evaluator
from shopper_intent.models.registry import ModelRegistry
from shopper_intent.models.trainer import Trainer, TrainingData, TrainResult
from shopper_intent.models.validation import CrossValidation


def _train(uri: str, pipeline: Pipeline, data: TrainingData) -> TrainResult:
    trainer = Trainer(
        tracking_uri=uri,
        experiment_name="test-experiment",
        evaluator=Evaluator(),
        cross_validation=CrossValidation(3, "average_precision", seed=0),
        tags={"data_md5": "abc123"},
    )
    return trainer.train_and_log("logistic_regression", pipeline, {"C": 1.0}, data)


def test_train_and_log_records_params_tags_and_metrics(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(mlflow_tracking_uri, logistic_pipeline, tiny_data)

    run = mlflow.get_run(result.run_id)
    assert run.data.params["seed"] == "0"
    assert run.data.params["cv_folds"] == "3"
    assert run.data.tags["data_md5"] == "abc123"
    assert run.data.metrics["cv_average_precision"] == pytest.approx(result.cv_score)
    assert run.data.metrics["average_precision"] == pytest.approx(
        result.metrics["average_precision"]
    )


def test_registry_registers_run_and_points_alias(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(mlflow_tracking_uri, logistic_pipeline, tiny_data)
    registry = ModelRegistry(mlflow_tracking_uri, "test-model", "champion")

    registry.promote(registry.register(result.run_id))

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    assert client.get_model_version_by_alias("test-model", "champion").run_id == result.run_id
