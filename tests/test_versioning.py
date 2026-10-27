"""Testes da leitura da versão do dataset a partir do arquivo .dvc."""

from pathlib import Path

from shopper_intent.data.versioning import dvc_file_for, read_dvc_md5


def test_dvc_file_for_appends_dvc_extension() -> None:
    assert dvc_file_for(Path("data/raw/sessions.csv")) == Path("data/raw/sessions.csv.dvc")


def test_read_dvc_md5_returns_hash_of_tracked_file(tmp_path: Path) -> None:
    dvc_file = tmp_path / "sessions.csv.dvc"
    dvc_file.write_text("outs:\n- md5: abc123\n  size: 10\n  hash: md5\n  path: sessions.csv\n")

    assert read_dvc_md5(dvc_file) == "abc123"
