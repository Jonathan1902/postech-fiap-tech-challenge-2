"""Construção do ColumnTransformer de pré-processamento das features."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureBuilder:
    """Monta o ColumnTransformer: one-hot para categóricas e escala para numéricas.

    Colunas fora das listas (ex.: ``Weekend``, já 0/1) passam sem alteração.
    """

    def __init__(self, numeric_columns: list[str], categorical_columns: list[str]) -> None:
        """Guarda as colunas numéricas e categóricas do dataset."""
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

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
