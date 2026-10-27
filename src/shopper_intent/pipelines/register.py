"""Entrypoint do estágio `register` do DVC: melhor run -> Model Registry (alias champion)."""

import json
from pathlib import Path

from shopper_intent.config import Settings, load_params
from shopper_intent.models.registry import ModelRegistry


def main() -> None:
    """Escolhe o melhor run pela métrica primária, registra e aplica o alias."""
    params = load_params()
    settings = Settings()
    registry = ModelRegistry(
        settings.mlflow_tracking_uri, settings.registered_model_name, settings.model_alias
    )
    run_id = registry.best_run_id(
        settings.mlflow_experiment_name, params["train"]["primary_metric"]
    )
    version = registry.register(run_id)
    registry.promote(version)
    output = Path(params["register"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"model": registry.model_name, "version": version, "alias": registry.alias}
    output.write_text(json.dumps({**summary, "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
