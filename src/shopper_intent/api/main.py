"""API FastAPI de inferência: /health e /predict."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status

from shopper_intent.api.model_loader import LoadedModel, load_champion
from shopper_intent.api.schemas import HealthResponse, PredictionResponse, SessionFeatures
from shopper_intent.config import Settings

ModelLoader = Callable[[], LoadedModel]

logger = logging.getLogger(__name__)
router = APIRouter()


def _try_load(load_model: ModelLoader) -> LoadedModel | None:
    """Carrega o modelo; em caso de falha, registra o erro e segue sem modelo (503)."""
    try:
        return load_model()
    except Exception:
        logger.exception("Não foi possível carregar o modelo do Registry.")
        return None


def get_model(request: Request) -> LoadedModel:
    """Dependência: devolve o modelo carregado ou responde 503 se não houver."""
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo indisponível.")
    return model


@router.get("/health")
def health(request: Request, response: Response) -> HealthResponse:
    """Indica se a API está no ar e qual versão do modelo está servindo."""
    model: LoadedModel | None = request.app.state.model
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", model_version=None)
    return HealthResponse(status="ok", model_version=model.version)


@router.post("/predict")
def predict(
    session: SessionFeatures, model: Annotated[LoadedModel, Depends(get_model)]
) -> PredictionResponse:
    """Recebe as features da sessão e devolve a probabilidade de compra."""
    probability = model.purchase_probability(session.to_frame())
    return PredictionResponse(
        purchase_probability=probability,
        will_purchase=probability >= model.threshold,
        threshold=model.threshold,
        model_version=model.version,
    )


def create_app(load_model: ModelLoader) -> FastAPI:
    """Cria a aplicação; o modelo é carregado uma vez, na inicialização."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = _try_load(load_model)
        yield

    app = FastAPI(title="Shopper Intent API", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app(lambda: load_champion(Settings()))
