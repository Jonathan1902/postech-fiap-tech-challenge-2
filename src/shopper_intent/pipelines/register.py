"""Entrypoint do estágio `register` do DVC: run escolhido no train -> Registry (alias)."""

import json
from pathlib import Path

import mlflow

from shopper_intent.config import Settings, load_params
from shopper_intent.files import write_json
from shopper_intent.models.registry import ModelRegistry
from shopper_intent.models.validation import cv_metric_name


def read_best_run_id(path: Path) -> str:
    """Lê o run_id do melhor candidato gravado pelo estágio train."""
    return str(json.loads(path.read_text(encoding="utf-8"))["run_id"])


def main() -> None:
    """Registra o run escolhido pelo train, promove se não for pior e grava um resumo."""
    params = load_params()
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    registry = ModelRegistry(
        settings.registered_model_name,
        settings.model_alias,
        metric=cv_metric_name(params["train"]["primary_metric"]),
    )
    run_id = read_best_run_id(Path(params["train"]["best_run_path"]))
    version = registry.register(run_id)
    summary = {
        "model": registry.model_name,
        "version": version,
        "alias": registry.alias,
        "promoted": registry.promote(version),
        "run_id": run_id,
    }
    write_json(Path(params["register"]["output_path"]), summary)


if __name__ == "__main__":
    main()
