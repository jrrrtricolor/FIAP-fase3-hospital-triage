"""Testes essenciais da API de triagem."""

from pathlib import Path

from fastapi.testclient import TestClient

from ml_prep_kit import ModelPredictor
from src.hospital_triage.api import create_app, load_model


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


def test_bundled_onnx_model_serves_real_prediction(monkeypatch) -> None:
    """Carrega o artefato real usado pela imagem e executa uma inferência."""
    model_path = (
        Path(__file__).resolve().parents[1]
        / "model/hospital_triage_model.onnx"
    )
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    real_predictor, model_version = load_model()
    real_app = create_app(
        predictor=real_predictor,
        model_version=model_version,
        model_loader=None,
    )

    with TestClient(real_app) as real_client:
        response = real_client.post(
            "/predict",
            json={"clinical_text": "Severe chest pain and shortness of breath."},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["target"] in {"normal", "atencao", "urgente"}
    assert set(payload["probabilities"]) == {"normal", "atencao", "urgente"}
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-5
    assert payload["model_version"] == "onnx-bundled-v1"
