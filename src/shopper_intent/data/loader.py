"""Leitura do dataset bruto de sessões de navegação."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetLoader:
    """Carrega o CSV bruto do Online Shoppers Intention."""

    path: Path

    def load(self) -> pd.DataFrame:
        """Lê o CSV e devolve o DataFrame sem nenhuma transformação."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset não encontrado: {self.path} (rode `dvc pull`).")
        return pd.read_csv(self.path)
