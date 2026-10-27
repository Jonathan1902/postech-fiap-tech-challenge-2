"""Configurações do projeto: variáveis de ambiente (.env) e parâmetros (params.yaml)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PARAMS_PATH = Path("params.yaml")


class Settings(BaseSettings):
    """Configurações lidas de variáveis de ambiente e do arquivo .env.

    Variáveis de ambiente têm precedência sobre o .env; os nomes não diferenciam
    maiúsculas de minúsculas (``MLFLOW_TRACKING_URI`` -> ``mlflow_tracking_uri``).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "shopper-intent"
    registered_model_name: str = "shopper-intent-model"
    model_alias: str = "champion"


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Lê o arquivo params.yaml e devolve os parâmetros como dicionário.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se o conteúdo não for um mapeamento YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {path}")
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise ValueError(f"{path} deve conter um mapeamento YAML na raiz.")
    return params
