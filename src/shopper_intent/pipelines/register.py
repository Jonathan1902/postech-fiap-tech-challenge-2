"""Entrypoint do estágio `register` do DVC: run escolhido no train -> Registry (alias)."""

import json
from pathlib import Path

from shopper_intent.config import Settings, load_params
from shopper_intent.models.registry import ModelRegistry


def read_best_run_id(path: Path) -> str:
    """Lê o run_id do melhor candidato gravado pelo estágio train."""
    return str(json.loads(path.read_text(encoding="utf-8"))["run_id"])


def main() -> None:
    """Registra o run escolhido pelo train, aplica o alias e grava um resumo em JSON."""
    params = load_params()
    settings = Settings()
    registry = ModelRegistry(
        settings.mlflow_tracking_uri, settings.registered_model_name, settings.model_alias
    )
    run_id = read_best_run_id(Path(params["train"]["best_run_path"]))
    version = registry.register(run_id)
    registry.promote(version)
    output = Path(params["register"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"model": registry.model_name, "version": version, "alias": registry.alias}
    output.write_text(json.dumps({**summary, "run_id": run_id}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
