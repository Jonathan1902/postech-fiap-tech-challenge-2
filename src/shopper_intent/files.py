"""Gravação dos artefatos dos estágios, criando a pasta de destino quando preciso."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: Path) -> Path:
    """Cria a pasta de destino do arquivo e devolve o próprio caminho."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Grava o dicionário em JSON indentado."""
    ensure_parent(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Grava o DataFrame em CSV, sem o índice."""
    df.to_csv(ensure_parent(path), index=False)
