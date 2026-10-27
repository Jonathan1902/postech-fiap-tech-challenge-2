"""Entrypoint do estágio `preprocess` do DVC: raw -> processed."""

from pathlib import Path

import pandas as pd

from shopper_intent.config import load_params
from shopper_intent.data.loader import DatasetLoader
from shopper_intent.data.preprocess import Preprocessor, StratifiedSplitter


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """Grava o DataFrame em CSV, criando a pasta de destino se preciso."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Carrega o CSV bruto, limpa, faz o split e grava train/test em data/processed."""
    params = load_params()
    data = params["data"]
    preprocessor = Preprocessor()
    raw = DatasetLoader(Path(data["raw_path"])).load()
    clean = preprocessor.cast_types(preprocessor.remove_duplicates(raw))
    splitter = StratifiedSplitter(params["split"]["test_size"], params["seed"], data["target"])
    train, test = splitter.split(clean)
    _write_csv(train, Path(data["train_path"]))
    _write_csv(test, Path(data["test_path"]))


if __name__ == "__main__":
    main()
