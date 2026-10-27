"""Construção do ColumnTransformer de pré-processamento das features."""

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureBuilder:
    """Monta o ColumnTransformer: one-hot para categóricas e escala para numéricas.

    Colunas fora das listas (ex.: ``Weekend``, já 0/1) passam sem alteração.
    """

    numeric_columns: list[str]
    categorical_columns: list[str]

    def build(self) -> ColumnTransformer:
        """Devolve o ColumnTransformer pronto para compor um Pipeline."""
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.numeric_columns),
                ("categorical", encoder, self.categorical_columns),
            ],
            remainder="passthrough",
        )
