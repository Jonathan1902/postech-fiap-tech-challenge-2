"""Entrypoint do estágio `train` do DVC: processed -> candidatos logados no MLflow."""

from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from shopper_intent.config import Settings, load_params
from shopper_intent.data.preprocess import split_features_target
from shopper_intent.data.versioning import dvc_file_for, read_dvc_md5
from shopper_intent.features.build_features import FeatureBuilder
from shopper_intent.files import ensure_parent, write_json
from shopper_intent.models.evaluator import Evaluator
from shopper_intent.models.factory import ModelFactory
from shopper_intent.models.trainer import Trainer, TrainingData, TrainResult
from shopper_intent.models.validation import CrossValidation


def _load_training_data(data_params: dict[str, Any]) -> TrainingData:
    """Lê train.csv e test.csv e separa features e alvo."""
    target = data_params["target"]
    x_train, y_train = split_features_target(pd.read_csv(data_params["train_path"]), target)
    x_test, y_test = split_features_target(pd.read_csv(data_params["test_path"]), target)
    return TrainingData(x_train, y_train, x_test, y_test)


def _build_trainer(params: dict[str, Any], experiment_name: str) -> Trainer:
    """Monta o Trainer com avaliador, validação cruzada e a versão do dataset como tag."""
    train_params = params["train"]
    data_md5 = read_dvc_md5(dvc_file_for(Path(params["data"]["raw_path"])))
    return Trainer(
        experiment_name=experiment_name,
        evaluator=Evaluator(train_params["threshold"]),
        cross_validation=CrossValidation(
            train_params["cv_folds"], train_params["primary_metric"], params["seed"]
        ),
        tags={"data_md5": data_md5},
    )


def _train_candidates(
    params: dict[str, Any], trainer: Trainer, data: TrainingData
) -> list[TrainResult]:
    """Treina todos os candidatos de params.yaml e loga cada um no MLflow."""
    candidates = params["train"]["models"]
    features = params["features"]
    preprocessor = FeatureBuilder(features["numeric"], features["categorical"]).build()
    factory = ModelFactory(candidates, params["seed"])
    return [
        trainer.train_and_log(name, factory.create(name, preprocessor), candidates[name], data)
        for name in factory.names()
    ]


def _save_best(best: TrainResult, train_params: dict[str, Any], cv_metric: str) -> None:
    """Grava o melhor Pipeline, suas métricas e o run_id que o estágio register promove."""
    joblib.dump(best.pipeline, ensure_parent(Path(train_params["model_path"])))
    metrics = {"best_model": best.name, cv_metric: best.cv_score, **best.metrics}
    write_json(Path(train_params["metrics_path"]), metrics)
    write_json(Path(train_params["best_run_path"]), {"model": best.name, "run_id": best.run_id})


def main() -> None:
    """Treina cada candidato, escolhe o melhor pela CV e grava modelo, métricas e run_id."""
    params = load_params()
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    trainer = _build_trainer(params, settings.mlflow_experiment_name)
    results = _train_candidates(params, trainer, _load_training_data(params["data"]))
    best = max(results, key=lambda result: result.cv_score)
    _save_best(best, params["train"], trainer.cross_validation.metric_name)


if __name__ == "__main__":
    main()
