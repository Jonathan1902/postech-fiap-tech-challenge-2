"""Leitura da versão (hash) de datasets rastreados pelo DVC."""

from pathlib import Path

import yaml


def dvc_file_for(data_path: Path) -> Path:
    """Devolve o caminho do arquivo .dvc que rastreia o dataset informado."""
    return data_path.with_name(f"{data_path.name}.dvc")


def read_dvc_md5(dvc_file: Path) -> str:
    """Devolve o md5 registrado no arquivo .dvc, que identifica a versão do dataset."""
    content = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return str(content["outs"][0]["md5"])
