"""API REST para inferência textual de urgência hospitalar."""

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, StringConstraints

from ml_prep_kit import ModelPredictor

from .training import (
    DEFAULT_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)

MAX_TEXT_LENGTH = 5_000
DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

ClinicalText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
    ),
]


class PredictionRequest(BaseModel):
    """Texto clínico em inglês recebido pela API."""

    clinical_text: ClinicalText


class PredictionResponse(BaseModel):
    """Classificação e probabilidades retornadas ao cliente."""

    target: Literal["normal", "atencao", "urgente"]
    probabilities: dict[str, float]
    model_version: str
    inference_time_ms: float


ModelLoader = Callable[[], tuple[ModelPredictor, str]]


def load_registered_model() -> tuple[ModelPredictor, str]:
    """Carrega uma única vez o modelo promovido no MLflow Registry."""
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_version = os.getenv("MODEL_VERSION", MODEL_ALIAS)
    predictor = ModelPredictor.from_mlflow(
        model_uri=model_uri,
        tracking_uri=tracking_uri,
    )
    return predictor, model_version


def create_app(
    predictor: ModelPredictor | None = None,
    model_version: str = "unavailable",
    model_loader: ModelLoader | None = load_registered_model,
) -> FastAPI:
    """Cria a aplicação e permite injetar o modelo nos testes."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Em produção, o modelo é carregado somente durante o startup.
        if application.state.predictor is None and model_loader is not None:
            loaded_predictor, loaded_version = model_loader()
            application.state.predictor = loaded_predictor
            application.state.model_version = loaded_version
        yield

    application = FastAPI(
        title="Hospital Triage API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.predictor = predictor
    application.state.model_version = model_version

    def require_predictor(request: Request) -> ModelPredictor:
        """Recusa inferências enquanto o modelo não estiver disponível."""
        loaded_predictor = request.app.state.predictor
        if loaded_predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo indisponível.",
            )
        return loaded_predictor

    @application.get("/health")
    def health() -> dict[str, str]:
        """Confirma que o processo da API está ativo."""
        return {"status": "ok"}

    @application.get("/ready")
    def ready(request: Request) -> dict[str, str | bool]:
        """Informa se o modelo já está carregado em memória."""
        is_ready = request.app.state.predictor is not None
        return {
            "ready": is_ready,
            "model_version": request.app.state.model_version,
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        loaded_predictor: Annotated[
            ModelPredictor,
            Depends(require_predictor),
        ],
        request: Request,
    ) -> PredictionResponse:
        """Classifica um texto sem armazená-lo ou escrevê-lo em logs."""
        started_at = time.perf_counter()
        texts = [payload.clinical_text]
        target = str(loaded_predictor.predict(texts)[0])
        scores = loaded_predictor.predict_proba(texts)[0]
        classes = loaded_predictor.model.classes_
        probabilities = {
            str(class_name): float(score)
            for class_name, score in zip(classes, scores, strict=True)
        }
        inference_time_ms = (time.perf_counter() - started_at) * 1_000
        return PredictionResponse(
            target=target,
            probabilities=probabilities,
            model_version=request.app.state.model_version,
            inference_time_ms=inference_time_ms,
        )

    return application


app = create_app()
