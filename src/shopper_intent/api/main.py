"""API FastAPI de inferência: /health e /predict."""

from typing import Any

from fastapi import FastAPI

app = FastAPI(title="Shopper Intent API")


@app.get("/health")
def health() -> dict[str, str]:
    """Indica se a API está no ar."""
    raise NotImplementedError


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, float]:
    """Recebe as features da sessão e devolve a probabilidade de compra."""
    raise NotImplementedError
