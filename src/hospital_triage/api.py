"""API REST para inferência textual de urgência hospitalar."""

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, StringConstraints

from ml_prep_kit import ModelPredictor

from .config.api_logging_middleware import LoggingMiddleware
from .config.logging_config import setup_api_logger
from .constants import TARGET_ORDER
from .prometheus.metrics import (
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_DURATION,
    PREDICTIONS_TOTAL,
)

MAX_TEXT_LENGTH = 5_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/hospital_triage_model.onnx"

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
LOGGER = setup_api_logger()


def load_model() -> tuple[ModelPredictor, str]:
    """Carrega uma única vez o modelo ONNX empacotado."""
    model_version = os.getenv("MODEL_VERSION", "onnx-bundled-v1")
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise FileNotFoundError(f"Artefato de inferência ausente: {model_path}.")
    if model_path.suffix != ".onnx":
        raise ValueError("MODEL_PATH deve apontar para um artefato .onnx.")
    predictor = ModelPredictor.from_onnx(
        model_path,
        classes=sorted(TARGET_ORDER),
    )
    return predictor, model_version


def create_app(
    predictor: ModelPredictor | None = None,
    model_version: str = "unavailable",
    model_loader: ModelLoader | None = load_model,
) -> FastAPI:
    """Cria a aplicação e permite injetar o modelo nos testes."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Em produção, o modelo é carregado somente durante o startup.
        if application.state.predictor is None and model_loader is not None:
            loaded_predictor, loaded_version = model_loader()
            application.state.predictor = loaded_predictor
            application.state.model_version = loaded_version
        MODEL_INFO.labels(version=application.state.model_version).set(1)
        yield

    application = FastAPI(
        title="Hospital Triage API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.predictor = predictor
    application.state.model_version = model_version
    if predictor is not None:
        MODEL_INFO.labels(version=model_version).set(1)
    application.add_middleware(LoggingMiddleware)
    Instrumentator().instrument(application).expose(
        application,
        endpoint="/metrics",
        summary="Métricas de desempenho da API para Prometheus",
    )

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

        LOGGER.info(
            "predicao_gerada",
            extra={
                "duracao_ms": inference_time_ms,
                "confianca": max(probabilities.values()),
                "probabilidades": probabilities,
            },
        )

        PREDICTION_DURATION.observe(inference_time_ms / 1_000)
        PREDICTIONS_TOTAL.inc()
        PREDICTION_CONFIDENCE.observe(max(probabilities.values()))

        return PredictionResponse(
            target=target,
            probabilities=probabilities,
            model_version=request.app.state.model_version,
            inference_time_ms=inference_time_ms,
        )

    return application


app = create_app()
