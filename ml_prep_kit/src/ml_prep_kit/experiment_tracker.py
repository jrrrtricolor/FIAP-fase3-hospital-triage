"""Registro reutilizável de experimentos com MLflow."""

import io
import logging
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Centraliza operações comuns de tracking com MLflow.

    Use esta classe para configurar o experimento uma única vez e registrar
    parâmetros, métricas, artefatos e modelos de forma padronizada.

    Exemplo:
        tracker = ExperimentTracker(
            experiment_name="experimento-classificacao-textual",
            tracking_uri="sqlite:///mlflow.db",
        )

        run_id = tracker.log_training_run(
            run_name="logistic_regression",
            model=pipeline,
            parameters={"model_name": "logistic_regression"},
            metrics={"roc_auc": 0.82},
        )
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """Configura o experimento usado pelos registros.

        Exemplo:
            tracker = ExperimentTracker(
                experiment_name="experimento-classificacao-textual",
                tracking_uri="sqlite:///mlflow.db",
            )
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        model_name: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo treinado.

        Exemplo:
            run_id = tracker.log_training_run(
                run_name="random_forest",
                model=pipeline,
                parameters={"n_estimators": 100},
                metrics={"f1": 0.74},
                artifacts=["model/report.html"],
            )
        """
        logger.info(
            "Registrando execução no MLflow.",
            extra={
                "evento": "registro_sklearn_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Saving scikit-learn models.*",
                )
                with redirect_stdout(io.StringIO()):
                    with redirect_stderr(io.StringIO()):
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            name=model_name,
                            registered_model_name=registered_model_name,
                        )

            logger.info(
                "Registro concluído no MLflow.",
                extra={
                    "evento": "registro_sklearn_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
            )

            return run.info.run_id

    def log_pytorch_training_run(
        self,
        run_name: str,
        model: Any,
        parameters: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[str | Path] | None = None,
        input_example: Any | None = None,
        model_name: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """Registra parâmetros, métricas, artefatos e modelo PyTorch.

        Exemplo:
            run_id = tracker.log_pytorch_training_run(
                run_name="text_multiclass_classifier",
                model=torch_model,
                parameters={"epochs": 10, "learning_rate": 0.001},
                metrics={"roc_auc": 0.88},
                input_example=X_valid_ready[:5],
            )
        """
        logger.info(
            "Registrando rede neural no MLflow.",
            extra={
                "evento": "registro_pytorch_mlflow_iniciado",
                "nome_execucao": run_name,
                "nome_experimento": self.experiment_name,
                "quantidade_artefatos": len(artifacts or []),
            },
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(self._stringify_values(parameters))
            mlflow.log_metrics(metrics)

            for artifact in artifacts or []:
                mlflow.log_artifact(str(artifact))

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given buffer is not writable.*",
                )
                with redirect_stdout(io.StringIO()):
                    with redirect_stderr(io.StringIO()):
                        mlflow.pytorch.log_model(
                            pytorch_model=model,
                            name=model_name,
                            input_example=input_example,
                            serialization_format="pt2",
                            registered_model_name=registered_model_name,
                        )

            logger.info(
                "Rede neural registrada no MLflow.",
                extra={
                    "evento": "registro_pytorch_mlflow_concluido",
                    "nome_execucao": run_name,
                    "execucao_id": run.info.run_id,
                    "nome_experimento": self.experiment_name,
                },
            )

            return run.info.run_id

    def promote_latest_model_version(
        self,
        registered_model_name: str,
        alias: str = "production",
    ) -> str:
        """Promove a versão mais recente do modelo registrado.

        Exemplo:
            version = tracker.promote_latest_model_version(
                registered_model_name="hospital_triage_text_classifier",
                alias="champion",
            )
        """
        # Buscar versões registradas para o modelo informado.
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name = '{registered_model_name}'"
        )

        # Interromper quando o modelo ainda não existir no registro.
        if not versions:
            raise ValueError(
                f"Modelo registrado não encontrado: {registered_model_name}."
            )

        # Selecionar a versão mais recente do modelo.
        latest_version = max(
            versions,
            key=lambda version: int(version.version),
        )

        # Aplicar o alias na versão mais recente.
        client.set_registered_model_alias(
            registered_model_name,
            alias,
            latest_version.version,
        )
        logger.info(
            "Modelo promovido para produção.",
            extra={
                "evento": "modelo_promovido_registry",
                "nome_modelo_registrado": registered_model_name,
                "versao": latest_version.version,
                "alias": alias,
            },
        )

        return str(latest_version.version)

    def _stringify_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Converte valores complexos para texto aceito pelo MLflow."""
        formatted = {}

        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                formatted[key] = value
            else:
                formatted[key] = str(value)

        return formatted
