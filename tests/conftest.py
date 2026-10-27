"""Fixtures compartilhadas: dados pequenos e um MLflow isolado em SQLite temporário."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from shopper_intent.features.build_features import FeatureBuilder
from shopper_intent.models.factory import ModelFactory
from shopper_intent.models.trainer import TrainingData

SAMPLE_ROWS = 60
TRAIN_ROWS = 40


@pytest.fixture
def tiny_data() -> TrainingData:
    """Dataset sintético em que PageValues positivo indica compra."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "PageValues": rng.normal(size=SAMPLE_ROWS),
            "Browser": rng.integers(1, 3, size=SAMPLE_ROWS),
        }
    )
    y = (x["PageValues"] > 0).astype(int)
    return TrainingData(
        x.iloc[:TRAIN_ROWS], y.iloc[:TRAIN_ROWS], x.iloc[TRAIN_ROWS:], y.iloc[TRAIN_ROWS:]
    )


@pytest.fixture
def logistic_pipeline() -> Pipeline:
    """Pipeline de regressão logística sobre as colunas de tiny_data."""
    preprocessor = FeatureBuilder(["PageValues"], ["Browser"]).build()
    factory = ModelFactory({"logistic_regression": {"C": 1.0}}, seed=0)
    return factory.create("logistic_regression", preprocessor)


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """URI de um MLflow em SQLite temporário; artefatos ficam dentro de tmp_path."""
    monkeypatch.chdir(tmp_path)
    return f"sqlite:///{tmp_path / 'mlflow.db'}"
