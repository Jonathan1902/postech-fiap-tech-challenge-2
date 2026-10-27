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

MODEL_NAME = "test-model"
CV_METRIC = "cv_average_precision"


def _train(pipeline: Pipeline, data: TrainingData) -> TrainResult:
    trainer = Trainer(
        experiment_name="test-experiment",
        evaluator=Evaluator(),
        cross_validation=CrossValidation(3, "average_precision", seed=0),
        tags={"data_md5": "abc123"},
    )
    return trainer.train_and_log("logistic_regression", pipeline, {"C": 1.0}, data)


def _registry() -> ModelRegistry:
    return ModelRegistry(MODEL_NAME, "champion", metric=CV_METRIC)


def _champion_run_id() -> str:
    return MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").run_id


def test_train_and_log_records_params_tags_and_metrics(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(logistic_pipeline, tiny_data)

    run = mlflow.get_run(result.run_id)
    assert run.data.params["seed"] == "0"
    assert run.data.params["cv_folds"] == "3"
    assert run.data.tags["data_md5"] == "abc123"
    assert run.data.metrics[CV_METRIC] == pytest.approx(result.cv_score)
    assert run.data.metrics["average_precision"] == pytest.approx(
        result.metrics["average_precision"]
    )
    confusion = mlflow.artifacts.load_dict(f"runs:/{result.run_id}/{CONFUSION_MATRIX_ARTIFACT}")
    assert sum(confusion.values()) == len(tiny_data.y_test)


def test_registry_registers_run_and_points_alias(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(logistic_pipeline, tiny_data)
    registry = _registry()

    assert registry.promote(registry.register(result.run_id))
    assert _champion_run_id() == result.run_id


def test_register_is_idempotent_for_the_same_run(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(logistic_pipeline, tiny_data)
    registry = _registry()

    assert registry.register(result.run_id) == registry.register(result.run_id) == "1"


def test_promote_keeps_champion_when_new_version_is_worse(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    champion = _train(logistic_pipeline, tiny_data)
    challenger = _train(logistic_pipeline, tiny_data)
    MlflowClient().log_metric(challenger.run_id, CV_METRIC, champion.cv_score - 0.1)
    registry = _registry()
    registry.promote(registry.register(champion.run_id))

    promoted = registry.promote(registry.register(challenger.run_id))

    assert not promoted
    assert _champion_run_id() == champion.run_id


def test_load_champion_returns_promoted_version_and_threshold(
    mlflow_tracking_uri: str, logistic_pipeline: Pipeline, tiny_data: TrainingData
) -> None:
    result = _train(logistic_pipeline, tiny_data)
    registry = _registry()
    registry.promote(registry.register(result.run_id))
    settings = Settings(mlflow_tracking_uri=mlflow_tracking_uri, registered_model_name=MODEL_NAME)

    loaded = load_champion(settings)

    assert (loaded.version, loaded.threshold) == ("1", 0.5)
    assert 0.0 <= loaded.purchase_probability(tiny_data.x_test.head(1)) <= 1.0
