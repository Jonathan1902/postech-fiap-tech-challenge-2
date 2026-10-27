"""Testes de Settings e load_params."""

from pathlib import Path

import pytest

from shopper_intent.config import Settings, load_params


def test_settings_defaults_without_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.model_alias == "champion"


def test_settings_reads_env_file_and_env_var_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    (tmp_path / ".env").write_text(
        "MLFLOW_TRACKING_URI=http://from-file:5000\nMODEL_ALIAS=staging\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_ALIAS", "champion-env")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://from-file:5000"
    assert settings.model_alias == "champion-env"


def test_load_params_returns_mapping(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text("seed: 42\nsplit:\n  test_size: 0.2\n")

    assert load_params(path) == {"seed": 42, "split": {"test_size": 0.2}}


def test_load_params_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nope.yaml")


def test_load_params_non_mapping_raises(tmp_path: Path):
    path = tmp_path / "params.yaml"
    path.write_text("- a\n- b\n")

    with pytest.raises(ValueError):
        load_params(path)
