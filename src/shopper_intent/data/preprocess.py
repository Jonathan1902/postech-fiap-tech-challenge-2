"""Limpeza do dataset: duplicatas, tipos e codificação do alvo."""

import pandas as pd

TARGET_COLUMN = "Revenue"


class Preprocessor:
    """Aplica a limpeza que roda antes do split treino/teste."""

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas duplicadas do DataFrame."""
        raise NotImplementedError

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte Weekend e Revenue para inteiro (0/1)."""
        raise NotImplementedError


class StratifiedSplitter:
    """Divide o dataset em treino e teste de forma estratificada pelo alvo."""

    def __init__(self, test_size: float, seed: int) -> None:
        """Guarda a proporção de teste e a seed de reprodutibilidade."""
        self.test_size = test_size
        self.seed = seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devolve os DataFrames de treino e de teste, mantendo a proporção do alvo."""
        raise NotImplementedError
