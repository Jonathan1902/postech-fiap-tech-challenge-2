"""Construção do ColumnTransformer de pré-processamento das features."""

from sklearn.compose import ColumnTransformer


class FeatureBuilder:
    """Monta o ColumnTransformer: one-hot para categóricas e escala para numéricas."""

    def __init__(self, numeric_columns: list[str], categorical_columns: list[str]) -> None:
        """Guarda as colunas numéricas e categóricas do dataset."""
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def build(self) -> ColumnTransformer:
        """Devolve o ColumnTransformer pronto para compor um Pipeline."""
        raise NotImplementedError
