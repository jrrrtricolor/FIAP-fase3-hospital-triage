"""Testes do comando de treinamento do classificador textual."""

import json
import warnings

import joblib
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from ml_prep_kit import SQLiteDataFrameStore
from src.hospital_triage.training import (
    train_and_export,
    validate_evaluation_report,
    validate_registered_model,
)


def test_trains_and_exports_model_with_mlflow(tmp_path, monkeypatch) -> None:
    """Valida modelo, métricas e tracking a partir do SQLite preparado."""
    git_sha = "a" * 40
    rows = []
    class_texts = {
        "normal": "routine evaluation stable patient",
        "atencao": "moderate symptom medical attention",
        "urgente": "critical symptom immediate emergency",
    }

    # Cada classe aparece em treino e validação como no contrato oficial.
    for target, text in class_texts.items():
        for index in range(4):
            rows.append(
                {
                    "clinical_text": f"{text} train {index}",
                    "target": target,
                    "split": "train",
                    "dataset_version": "test-dataset-v1",
                }
            )
        for index in range(2):
            rows.append(
                {
                    "clinical_text": f"{text} validation {index}",
                    "target": target,
                    "split": "validation",
                    "dataset_version": "test-dataset-v1",
                }
            )

    database_path = tmp_path / "training_data.db"
    model_path = tmp_path / "model.joblib"
    onnx_path = tmp_path / "model.onnx"
    metrics_path = tmp_path / "metrics.json"
    optimization_path = tmp_path / "onnx_benchmark.json"
    mlflow_path = tmp_path / "mlflow.db"
    SQLiteDataFrameStore(database_path).save_dataframe(
        pd.DataFrame(rows),
        "training_data",
    )

    # Mantém também os artefatos internos do MLflow no diretório temporário.
    monkeypatch.chdir(tmp_path)
    report = train_and_export(
        database_path=database_path,
        model_path=model_path,
        metrics_path=metrics_path,
        tracking_uri=f"sqlite:///{mlflow_path}",
        git_sha=git_sha,
    )
    evaluated_report = validate_evaluation_report(metrics_path)
    registered_report = validate_registered_model(
        metrics_path=metrics_path,
        tracking_uri=f"sqlite:///{mlflow_path}",
    )

    saved_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    optimization_report = json.loads(optimization_path.read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="joblib.numpy_pickle",
        )
        saved_model = joblib.load(model_path)

    assert model_path.exists()
    assert onnx_path.exists()
    assert mlflow_path.exists()
    assert saved_report["mlflow_run_id"] == report["mlflow_run_id"]
    assert evaluated_report["mlflow_run_id"] == report["mlflow_run_id"]
    assert registered_report["mlflow_run_id"] == report["mlflow_run_id"]
    assert saved_report["dataset_version"] == "test-dataset-v1"
    assert saved_report["metrics"]["f1_macro"] == 1.0
    assert saved_report["registered_model_name"] == "hospital-triage-text-classifier"
    assert saved_report["registered_model_version"] == "1"
    assert saved_report["model_alias"] == "champion"
    assert saved_report["git_sha"] == git_sha
    assert optimization_report["technique"] == "ONNX Runtime"
    assert optimization_report["git_sha"] == git_sha
    assert optimization_report["comparison"]["prediction_agreement"] == 1.0
    assert optimization_report["f1_macro"] == 1.0
    assert saved_report["metrics"]["onnx_speedup"] > 0

    # ONNX e benchmark ficam ligados à mesma execução do modelo registrado.
    artifact_names = {
        artifact.path
        for artifact in MlflowClient().list_artifacts(report["mlflow_run_id"])
    }
    assert "model.onnx" in artifact_names
    assert "onnx_benchmark.json" in artifact_names

    # A versão do Registry deve apontar para o commit que gerou o modelo.
    registered_version = MlflowClient().get_model_version(
        name="hospital-triage-text-classifier",
        version="1",
    )
    assert registered_version.tags["git_sha"] == git_sha
    assert saved_model.predict(["critical emergency symptom"])[0] == "urgente"


def test_stops_before_fit_when_validation_split_is_missing(
    tmp_path,
) -> None:
    """Confirma o fail fast antes da criação de modelo ou run do MLflow."""
    database_path = tmp_path / "invalid_training_data.db"
    invalid_data = pd.DataFrame(
        {
            "clinical_text": ["stable", "moderate", "critical"],
            "target": ["normal", "atencao", "urgente"],
            "split": ["train", "train", "train"],
            "dataset_version": ["test-dataset-v1"] * 3,
        }
    )
    SQLiteDataFrameStore(database_path).save_dataframe(
        invalid_data,
        "training_data",
    )

    model_path = tmp_path / "model.joblib"
    mlflow_path = tmp_path / "mlflow.db"
    with pytest.raises(ValueError, match="validation"):
        train_and_export(
            database_path=database_path,
            model_path=model_path,
            metrics_path=tmp_path / "metrics.json",
            tracking_uri=f"sqlite:///{mlflow_path}",
        )

    assert not model_path.exists()
    assert not mlflow_path.exists()


def test_stops_before_data_access_when_git_sha_is_invalid(tmp_path) -> None:
    """Aplica fail fast quando a identificação do commit é inválida."""
    with pytest.raises(ValueError, match="git_sha"):
        train_and_export(
            database_path=tmp_path / "missing.db",
            model_path=tmp_path / "model.joblib",
            metrics_path=tmp_path / "metrics.json",
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            git_sha="sha-invalido",
        )

    assert not (tmp_path / "mlflow.db").exists()
