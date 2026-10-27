"""Limpeza do dataset (duplicatas, tipos) e split estratificado treino/teste."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Revenue"
BOOLEAN_COLUMNS = ("Weekend", TARGET_COLUMN)


class Preprocessor:
    """Aplica a limpeza que roda antes do split treino/teste."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas e converte as colunas booleanas para 0/1."""
        return self.cast_types(self.remove_duplicates(df))

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linhas duplicadas do DataFrame."""
        return df.drop_duplicates().reset_index(drop=True)

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte Weekend e Revenue para inteiro (0/1)."""
        return df.astype(dict.fromkeys(BOOLEAN_COLUMNS, int))


@dataclass(frozen=True)
class StratifiedSplitter:
    """Divide o dataset em treino e teste de forma estratificada pelo alvo."""

    test_size: float
    seed: int
    target: str = TARGET_COLUMN

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devolve os DataFrames de treino e de teste, mantendo a proporção do alvo."""
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df[self.target],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features (X) do alvo (y)."""
    return df.drop(columns=[target]), df[target]
