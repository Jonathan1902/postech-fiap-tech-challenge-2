"""Testes do Trainer e do ModelRegistry contra um MLflow em SQLite temporário."""

import mlflow
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from shopper_intent.api.model_loader import load_champion
from shopper_intent.config import Settings
from shopper_intent.models.evaluator import Evaluator
from shopper_intent.models.registry import ModelRegistry
from shopper_intent.models.trainer import (
    CONFUSION_MATRIX_ARTIFACT,
    Trainer,
    TrainingData,
    TrainResult,
)
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
    confusion = mlflow.artifacts.load_dict(f"runs:/{result.run_id}/{CONFUSION_MATRIX_ARTIFACT}")
    assert sum(confusion.values()) == len(tiny_data.y_test)


def test_registry_registers_run_and_points_alias(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(mlflow_tracking_uri, logistic_pipeline, tiny_data)
    registry = ModelRegistry(mlflow_tracking_uri, "test-model", "champion")

    registry.promote(registry.register(result.run_id))

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    assert client.get_model_version_by_alias("test-model", "champion").run_id == result.run_id


def test_load_champion_returns_promoted_version_and_threshold(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(mlflow_tracking_uri, logistic_pipeline, tiny_data)
    registry = ModelRegistry(mlflow_tracking_uri, "test-model", "champion")
    registry.promote(registry.register(result.run_id))
    settings = Settings(mlflow_tracking_uri=mlflow_tracking_uri, registered_model_name="test-model")

    loaded = load_champion(settings)

    assert (loaded.version, loaded.threshold) == ("1", 0.5)
    assert 0.0 <= loaded.purchase_probability(tiny_data.x_test.head(1)) <= 1.0
