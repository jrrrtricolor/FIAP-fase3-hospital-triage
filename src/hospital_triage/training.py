"""Treina e exporta o modelo selecionado para triagem hospitalar."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from ml_prep_kit import (
    ExperimentTracker,
    ModelEvaluator,
    ModelFactory,
    SQLiteDataFrameStore,
)

from .data_preparation import TARGET_ORDER

MODEL_NAME = "tfidf-logistic-regression"
EXPERIMENT_NAME = "hospital-triage-training"
REGISTERED_MODEL_NAME = "hospital-triage-text-classifier"
MODEL_ALIAS = "champion"
RANDOM_STATE = 42
MINIMUM_F1_MACRO = 0.55
MINIMUM_URGENT_RECALL = 0.55
GIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data/processed/training_data.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model/hospital_triage_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "model/training_metrics.json"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def build_model() -> Pipeline:
    """Cria o baseline escolhido no notebook de comparação."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10_000,
        # Regex ASCII equivalente para o corpus em inglês e compatível com RE2.
        token_pattern=r"[A-Za-z0-9_]{2,}",
    )
    return ModelFactory(random_state=RANDOM_STATE).create_pipeline(
        preprocessor=vectorizer,
        problem_type="classification",
        model_name="logistic_regression",
        parameters={
            "max_iter": 1_000,
            "class_weight": "balanced",
        },
    )


def train_and_export(
    database_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    tracking_uri: str,
    git_sha: str = "local",
) -> dict[str, object]:
    """Treina, otimiza com ONNX e registra os artefatos no MLflow."""
    _validate_git_sha(git_sha)
    store = SQLiteDataFrameStore(database_path)
    data = store.load_dataframe(
        table_name="training_data",
        columns=["clinical_text", "target", "split", "dataset_version"],
    )
    dataset_version = _validate_training_data(data)

    train_data = data.loc[data["split"] == "train"]
    validation_data = data.loc[data["split"] == "validation"]
    model = build_model()

    # O ajuste usa somente treino; validação permanece fora do fit.
    training_start = time.perf_counter()
    model.fit(train_data["clinical_text"], train_data["target"])
    training_seconds = time.perf_counter() - training_start

    predictions = model.predict(validation_data["clinical_text"])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(
        validation_data["target"],
        predictions,
    )
    by_class = evaluator.evaluate_classification_by_class(
        validation_data["target"],
        predictions,
    ).set_index("class")
    metrics.update(
        {
            "urgent_recall": float(by_class.loc["urgente", "recall"]),
            "training_seconds": training_seconds,
        }
    )
    _validate_model_metrics(metrics)

    model_output = Path(model_path)
    metrics_output = Path(metrics_path)
    onnx_output = model_output.with_suffix(".onnx")
    optimization_output = metrics_output.with_name("onnx_benchmark.json")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output)
    comparison = _benchmark_onnx(
        model,
        validation_data["clinical_text"].tolist(),
        model_output,
        onnx_output,
    )
    if comparison["prediction_agreement"] != 1.0:
        raise ValueError("O modelo ONNX alterou as classes previstas.")
    if comparison["speedup"] <= 1 and comparison["size_reduction_percent"] <= 0:
        raise ValueError("O modelo ONNX não apresentou ganho.")
    metrics.update(
        {
            "latency_ms_per_record": comparison["original_latency_ms_per_record"],
            "onnx_latency_ms_per_record": comparison["onnx_latency_ms_per_record"],
            "onnx_speedup": comparison["speedup"],
            "onnx_size_reduction_percent": comparison["size_reduction_percent"],
        }
    )

    optimization_report = {
        "technique": "ONNX Runtime",
        "dataset_version": dataset_version,
        "git_sha": git_sha,
        "validation_rows": len(validation_data),
        "f1_macro": metrics["f1_macro"],
        "urgent_recall": metrics["urgent_recall"],
        "comparison": comparison,
    }
    optimization_output.write_text(
        json.dumps(optimization_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    parameters = {
        "model_name": MODEL_NAME,
        "dataset_version": dataset_version,
        "vectorizer": "TF-IDF",
        "ngram_range": "(1, 2)",
        "max_features": 10_000,
        "token_pattern": r"[A-Za-z0-9_]{2,}",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "minimum_f1_macro": MINIMUM_F1_MACRO,
        "minimum_urgent_recall": MINIMUM_URGENT_RECALL,
        "git_sha": git_sha,
        "optimization": "ONNX Runtime",
    }

    # O framework centraliza o tracking e o registro do pipeline completo.
    tracker = ExperimentTracker(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=tracking_uri,
    )
    run_id = tracker.log_training_run(
        run_name=MODEL_NAME,
        model=model,
        parameters=parameters,
        metrics=metrics,
        artifacts=[onnx_output, optimization_output],
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    model_version = tracker.promote_latest_model_version(
        registered_model_name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        tags={"git_sha": git_sha},
    )

    report = {
        "model_name": MODEL_NAME,
        "dataset_version": dataset_version,
        "selection_metric": "f1_macro",
        "mlflow_run_id": run_id,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": model_version,
        "model_alias": MODEL_ALIAS,
        "git_sha": git_sha,
        "parameters": parameters,
        "metrics": metrics,
        "optimization": optimization_report,
    }

    metrics_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _benchmark_onnx(
    model: Pipeline,
    texts: list[str],
    model_path: Path,
    onnx_path: Path,
) -> dict[str, float | int]:
    """Exporta o ONNX e compara latência, predições e tamanho."""
    import numpy as np
    import onnxruntime as ort
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    converted = to_onnx(
        model,
        initial_types=[("clinical_text", StringTensorType([None, 1]))],
        options={"zipmap": False},
        target_opset=18,
    )
    onnx_path.write_bytes(converted.SerializeToString())

    ort.disable_telemetry_events()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_data = np.asarray(texts, dtype=object).reshape(-1, 1)

    started_at = time.perf_counter()
    original_predictions = np.asarray(model.predict(texts))
    original_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    onnx_predictions = np.asarray(session.run(None, {"clinical_text": input_data})[0])
    onnx_seconds = time.perf_counter() - started_at
    original_size, onnx_size = model_path.stat().st_size, onnx_path.stat().st_size
    return {
        "prediction_agreement": float(
            np.mean(original_predictions == onnx_predictions)
        ),
        "original_latency_ms_per_record": (original_seconds * 1_000 / len(texts)),
        "onnx_latency_ms_per_record": onnx_seconds * 1_000 / len(texts),
        "speedup": original_seconds / onnx_seconds,
        "original_size_bytes": original_size,
        "onnx_size_bytes": onnx_size,
        "size_reduction_percent": (1 - onnx_size / original_size) * 100,
    }


def _validate_training_data(data: pd.DataFrame) -> str:
    """Valida o contrato mínimo necessário antes do treinamento."""
    if data.empty:
        raise ValueError("A base de treinamento está vazia.")
    if data.isna().any().any():
        raise ValueError("A base de treinamento possui valores nulos.")

    versions = data["dataset_version"].unique()
    if len(versions) != 1:
        raise ValueError("A base deve possuir uma única versão.")

    for split in ("train", "validation"):
        split_data = data.loc[data["split"] == split]
        if split_data.empty:
            raise ValueError(f"O split {split} está vazio.")
        if set(split_data["target"]) != set(TARGET_ORDER):
            raise ValueError(f"O split {split} não possui todas as classes.")

    return str(versions[0])


def _validate_model_metrics(metrics: dict[str, float]) -> None:
    """Impede o registro de modelos abaixo dos critérios mínimos."""
    failed = []
    if metrics["f1_macro"] < MINIMUM_F1_MACRO:
        failed.append("f1_macro")
    if metrics["urgent_recall"] < MINIMUM_URGENT_RECALL:
        failed.append("urgent_recall")
    if failed:
        raise ValueError(f"Modelo abaixo dos critérios mínimos: {', '.join(failed)}.")


def _validate_git_sha(git_sha: str) -> None:
    """Aceita o SHA completo do GitHub ou o marcador de execução local."""
    if git_sha != "local" and not GIT_SHA_PATTERN.fullmatch(git_sha):
        raise ValueError("git_sha deve ser um SHA Git completo de 40 caracteres.")


def parse_args() -> argparse.Namespace:
    """Lê os caminhos usados pelo comando de treinamento."""
    parser = argparse.ArgumentParser(
        description="Treina o classificador textual de triagem hospitalar."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="Banco SQLite produzido pela preparação.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Arquivo joblib que receberá o pipeline treinado.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Arquivo JSON que receberá métricas e parâmetros.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="Backend usado pelo MLflow.",
    )
    parser.add_argument(
        "--git-sha",
        default=os.getenv("GITHUB_SHA", "local"),
        help="Commit associado à versão registrada no MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o treinamento e informa os artefatos produzidos."""
    args = parse_args()
    report = train_and_export(
        database_path=args.database,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        tracking_uri=args.tracking_uri,
        git_sha=args.git_sha,
    )
    print(
        f"Modelo {report['model_name']} treinado. "
        f"F1 macro: {report['metrics']['f1_macro']:.3f}."
    )
    print(f"Modelo salvo em {args.model_output}.")
    print(f"Ganho ONNX: {report['optimization']['comparison']['speedup']:.2f}x.")
    print(f"Métricas salvas em {args.metrics_output}.")


if __name__ == "__main__":
    main()
