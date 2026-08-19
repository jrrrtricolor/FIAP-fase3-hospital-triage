"""Testes essenciais da API de triagem."""

from fastapi.testclient import TestClient

from ml_prep_kit import ModelPredictor
from src.hospital_triage.api import create_app


class FakeTextClassifier:
    """Simula o modelo sem acessar o MLflow durante os testes."""

    classes_ = ("normal", "atencao", "urgente")

    def predict(self, texts: list[str]) -> list[str]:
        return ["urgente" for _ in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.05, 0.15, 0.80] for _ in texts]


predictor = ModelPredictor(model=FakeTextClassifier())
app = create_app(predictor=predictor, model_version="test-v1")
client = TestClient(app)


def test_predict_endpoint_success() -> None:
    """Valida uma predição realizada com sucesso."""
    response = client.post(
        "/predict",
        json={"clinical_text": "Severe chest pain."},
    )

    assert response.status_code == 200
    assert response.json()["target"] == "urgente"


def test_predict_endpoint_validation_error() -> None:
    """Rejeita texto clínico vazio."""
    response = client.post(
        "/predict",
        json={"clinical_text": ""},
    )

    assert response.status_code == 422
