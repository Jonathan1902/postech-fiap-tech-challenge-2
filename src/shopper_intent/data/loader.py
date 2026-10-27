"""Leitura do dataset bruto de sessões de navegação."""

from pathlib import Path

import pandas as pd


class DatasetLoader:
    """Carrega o CSV bruto do Online Shoppers Intention."""

    def __init__(self, path: Path) -> None:
        """Guarda o caminho do arquivo CSV."""
        self.path = path

    def load(self) -> pd.DataFrame:
        """Lê o CSV e devolve o DataFrame sem nenhuma transformação."""
        raise NotImplementedError
