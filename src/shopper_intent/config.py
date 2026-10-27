"""Configurações do projeto: variáveis de ambiente (.env) e parâmetros (params.yaml)."""

from pathlib import Path
from typing import Any


class Settings:
    """Agrupa as configurações lidas do ambiente e do arquivo .env."""

    def __init__(self, env_file: Path = Path(".env")) -> None:
        """Guarda o caminho do arquivo de ambiente a ser lido."""
        self.env_file = env_file

    def load(self) -> None:
        """Carrega as variáveis de ambiente e valida os valores obrigatórios."""
        raise NotImplementedError


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    """Lê o arquivo params.yaml e devolve os parâmetros como dicionário."""
    raise NotImplementedError
