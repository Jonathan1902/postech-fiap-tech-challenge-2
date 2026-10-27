"""Testes da API com um modelo falso injetado (sem MLflow)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shopper_intent.api.main import create_app
from shopper_intent.api.model_loader import LoadedModel
from shopper_intent.api.schemas import SessionFeatures

RAW_DATA = Path("data/raw/online_shoppers_intention.csv")

# Primeira sessão com compra do dataset original.
SESSION: dict[str, Any] = {
    "Administrative": 3,
    "Administrative_Duration": 87.83333333,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 27,
    "ProductRelated_Duration": 798.3333333,
    "BounceRates": 0.0,
    "ExitRates": 0.012643678,
    "PageValues": 22.9160357,
    "SpecialDay": 0.8,
    "Month": "Feb",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 3,
    "TrafficType": 1,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


class FakeClassifier:
    """Devolve sempre a mesma probabilidade e guarda a entrada recebida."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.received: pd.DataFrame | None = None

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.received = x
        return np.array([[1 - self.probability, self.probability]])


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier(probability=0.8)


@pytest.fixture
def client(classifier: FakeClassifier) -> Iterator[TestClient]:
    app = create_app(lambda: LoadedModel(classifier, version="7", threshold=0.5))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model() -> Iterator[TestClient]:
    def failing_loader() -> LoadedModel:
        raise LookupError("alias não encontrado")

    with TestClient(create_app(failing_loader)) as test_client:
        yield test_client


def test_health_reports_model_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "7"}


def test_predict_returns_probability_and_decision(client: TestClient) -> None:
    response = client.post("/predict", json=SESSION)

    assert response.status_code == 200
    assert response.json() == {
        "purchase_probability": pytest.approx(0.8),
        "will_purchase": True,
        "threshold": 0.5,
        "model_version": "7",
    }


def test_predict_sends_training_columns_to_model(
    client: TestClient, classifier: FakeClassifier
) -> None:
    client.post("/predict", json=SESSION)

    assert classifier.received is not None
    assert list(classifier.received.columns) == list(SESSION)
    assert classifier.received.loc[0, "Weekend"] == 0


@pytest.mark.parametrize(
    "override",
    [{"Month": "Junho"}, {"BounceRates": 1.5}, {"Revenue": True}, {"VisitorType": "Bot"}],
)
def test_predict_rejects_invalid_payload(client: TestClient, override: dict[str, Any]) -> None:
    response = client.post("/predict", json={**SESSION, **override})

    assert response.status_code == 422


def test_predict_rejects_missing_feature(client: TestClient) -> None:
    payload = {key: value for key, value in SESSION.items() if key != "PageValues"}

    assert client.post("/predict", json=payload).status_code == 422


def test_without_model_health_and_predict_return_503(client_without_model: TestClient) -> None:
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=SESSION).status_code == 503


def test_schema_columns_match_training_data() -> None:
    if not RAW_DATA.is_file():
        pytest.skip("Dataset ausente: rode `dvc pull`.")
    raw = pd.read_csv(RAW_DATA, nrows=1)
    expected = [column for column in raw.columns if column != "Revenue"]

    assert list(SessionFeatures(**SESSION).to_frame().columns) == expected
