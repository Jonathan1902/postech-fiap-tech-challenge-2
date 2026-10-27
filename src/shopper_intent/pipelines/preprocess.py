"""Entrypoint do estágio `preprocess` do DVC: raw -> processed."""

from pathlib import Path

from shopper_intent.config import load_params
from shopper_intent.data.loader import DatasetLoader
from shopper_intent.data.preprocess import Preprocessor, StratifiedSplitter
from shopper_intent.files import write_csv


def main() -> None:
    """Carrega o CSV bruto, limpa, faz o split e grava train/test em data/processed."""
    params = load_params()
    data = params["data"]
    clean = Preprocessor().clean(DatasetLoader(Path(data["raw_path"])).load())
    splitter = StratifiedSplitter(params["split"]["test_size"], params["seed"], data["target"])
    train, test = splitter.split(clean)
    write_csv(train, Path(data["train_path"]))
    write_csv(test, Path(data["test_path"]))


if __name__ == "__main__":
    main()
