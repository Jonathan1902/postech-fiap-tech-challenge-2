"""Catálogo de estimadores candidatos, cada um dentro de um Pipeline sklearn."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModelFactory:
    """Cria os estimadores candidatos a partir da configuração de params.yaml."""

    def __init__(self, candidates: dict[str, dict[str, Any]], seed: int) -> None:
        """Guarda a configuração dos candidatos e a seed."""
        self.candidates = candidates
        self.seed = seed

    def names(self) -> list[str]:
        """Lista os nomes dos candidatos configurados."""
        raise NotImplementedError

    def create(self, name: str, preprocessor: ColumnTransformer) -> Pipeline:
        """Monta o Pipeline (pré-processamento + estimador) do candidato informado."""
        raise NotImplementedError
