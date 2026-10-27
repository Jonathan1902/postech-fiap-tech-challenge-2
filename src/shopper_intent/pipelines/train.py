"""Entrypoint do estágio `train` do DVC: processed -> candidatos logados no MLflow."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from shopper_intent.config import Settings, load_params
from shopper_intent.data.preprocess import split_features_target
from shopper_intent.features.build_features import FeatureBuilder
from shopper_intent.models.evaluator import Evaluator
from shopper_intent.models.factory import ModelFactory
from shopper_intent.models.trainer import Trainer, TrainingData, TrainResult


def _load_training_data(data_params: dict[str, Any]) -> TrainingData:
    """Lê train.csv e test.csv e separa features e alvo."""
    target = data_params["target"]
    x_train, y_train = split_features_target(pd.read_csv(data_params["train_path"]), target)
    x_test, y_test = split_features_target(pd.read_csv(data_params["test_path"]), target)
    return TrainingData(x_train, y_train, x_test, y_test)


def _train_candidates(params: dict[str, Any], data: TrainingData) -> list[TrainResult]:
    """Treina todos os candidatos de params.yaml e loga cada um no MLflow."""
    settings = Settings()
    train_params = params["train"]
    features = params["features"]
    preprocessor = FeatureBuilder(features["numeric"], features["categorical"]).build()
    factory = ModelFactory(train_params["models"], params["seed"])
    trainer = Trainer(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
        Evaluator(train_params["threshold"]),
    )
    return [
        trainer.train_and_log(
            name, factory.create(name, preprocessor), train_params["models"][name], data
        )
        for name in factory.names()
    ]


def _save_best(best: TrainResult, model_path: Path, metrics_path: Path) -> None:
    """Grava o melhor Pipeline em disco e suas métricas em JSON."""
    for path in (model_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, model_path)
    metrics_path.write_text(json.dumps({"best_model": best.name, **best.metrics}, indent=2))


def main() -> None:
    """Treina cada candidato de params.yaml, loga no MLflow e grava reports/metrics.json."""
    params = load_params()
    train_params = params["train"]
    results = _train_candidates(params, _load_training_data(params["data"]))
    best = max(results, key=lambda result: result.metrics[train_params["primary_metric"]])
    _save_best(best, Path(train_params["model_path"]), Path(train_params["metrics_path"]))


if __name__ == "__main__":
    main()
