"""Fachada reutilizável para preparação e predição de modelos."""

from pathlib import Path
from typing import Any, Literal

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn


class ModelPredictor:
    """Combina um modelo e um preprocessador opcional para inferência.

    A classe não conhece features, classes ou arquitetura específica. O modelo
    precisa oferecer ``predict`` e, quando scores forem necessários,
    ``predict_proba``. Um pipeline Scikit-Learn completo pode ser usado sem
    preprocessador separado.

    Exemplo:
        predictor = ModelPredictor(model=pipeline)
        classes = predictor.predict(["texto para classificação"])
    """

    def __init__(
        self,
        model: Any | None = None,
        preprocessor: Any | None = None,
    ) -> None:
        """Guarda as dependências de inferência recebidas."""
        self.model = model
        self.preprocessor = preprocessor

    @classmethod
    def from_mlflow(
        cls,
        model_uri: str,
        flavor: Literal["sklearn", "pyfunc"] = "sklearn",
        tracking_uri: str | None = None,
        preprocessor: Any | None = None,
    ) -> "ModelPredictor":
        """Carrega um modelo por URI do MLflow.

        ``model_uri`` aceita URIs de runs e do Model Registry, incluindo aliases
        como ``models:/hospital_triage_text_classifier@champion``. Modelos
        PyTorch com tokenização própria devem ser registrados como wrapper
        Scikit-Learn ou modelo PyFunc para preservar o contrato de inferência.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        loaders = {
            "sklearn": mlflow.sklearn.load_model,
            "pyfunc": mlflow.pyfunc.load_model,
        }
        model = loaders[flavor](model_uri)
        return cls(model=model, preprocessor=preprocessor)

    def load_preprocessor(self, artifact_path: str | Path) -> None:
        """Carrega um preprocessador Joblib salvo localmente."""
        self.preprocessor = joblib.load(artifact_path)

    def prepare_input(self, X: Any) -> Any:
        """Aplica o preprocessador quando ele estiver configurado."""
        if self.preprocessor is None:
            return X

        if not hasattr(self.preprocessor, "transform"):
            raise TypeError("O preprocessador deve implementar transform().")

        return self.preprocessor.transform(X)

    def predict(self, X: Any) -> Any:
        """Prepara a entrada e retorna as classes ou valores previstos."""
        model = self._require_model()
        return model.predict(self.prepare_input(X))

    def predict_proba(self, X: Any) -> Any:
        """Prepara a entrada e retorna scores por classe."""
        model = self._require_model()

        if not hasattr(model, "predict_proba"):
            raise AttributeError("O modelo carregado não implementa predict_proba().")

        return model.predict_proba(self.prepare_input(X))

    def _require_model(self) -> Any:
        """Retorna o modelo configurado ou interrompe a inferência."""
        if self.model is None:
            raise ValueError("Nenhum modelo foi configurado para predição.")

        if not hasattr(self.model, "predict"):
            raise TypeError("O modelo deve implementar predict().")

        return self.model
