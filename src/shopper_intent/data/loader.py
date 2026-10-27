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
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset não encontrado: {self.path} (rode `dvc pull`).")
        return pd.read_csv(self.path)
