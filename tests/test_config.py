"""Testes de Settings e load_params."""

from pathlib import Path

import pytest

from shopper_intent.config import Settings, load_params

SETTINGS_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "REGISTERED_MODEL_NAME",
    "MODEL_ALIAS",
)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Roda o teste num diretório vazio e sem as variáveis de ambiente do projeto."""
    monkeypatch.chdir(tmp_path)
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_settings_defaults_without_env_file(isolated_env: Path) -> None:
    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.model_alias == "champion"


def test_settings_reads_env_file_and_env_var_wins(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated_env / ".env").write_text(
        "MLFLOW_TRACKING_URI=http://from-file:5000\nMODEL_ALIAS=staging\n"
    )
    monkeypatch.setenv("MODEL_ALIAS", "champion-env")

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://from-file:5000"
    assert settings.model_alias == "champion-env"


def test_load_params_returns_mapping(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("seed: 42\nsplit:\n  test_size: 0.2\n")

    assert load_params(path) == {"seed": 42, "split": {"test_size": 0.2}}


def test_load_params_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nope.yaml")


def test_load_params_non_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("- a\n- b\n")

    with pytest.raises(ValueError):
        load_params(path)
