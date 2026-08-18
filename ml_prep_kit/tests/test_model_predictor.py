"""Testes da fachada genérica de predição."""

import unittest
from unittest.mock import Mock, patch

from ml_prep_kit import ModelPredictor


class ModelPredictorTest(unittest.TestCase):
    """Valida injeção, preparação e carregamento de modelos."""

    def test_predict_applies_preprocessor_before_model(self) -> None:
        model = Mock()
        model.predict.return_value = ["urgente"]
        preprocessor = Mock()
        preprocessor.transform.return_value = [[1, 2, 3]]
        predictor = ModelPredictor(
            model=model,
            preprocessor=preprocessor,
        )

        result = predictor.predict(["achado crítico"])

        preprocessor.transform.assert_called_once_with(["achado crítico"])
        model.predict.assert_called_once_with([[1, 2, 3]])
        self.assertEqual(result, ["urgente"])

    def test_predict_proba_delegates_to_compatible_model(self) -> None:
        model = Mock()
        model.predict_proba.return_value = [[0.1, 0.2, 0.7]]
        predictor = ModelPredictor(model=model)

        result = predictor.predict_proba(["achado crítico"])

        model.predict_proba.assert_called_once_with(["achado crítico"])
        self.assertEqual(result, [[0.1, 0.2, 0.7]])

    def test_predict_raises_when_model_is_missing(self) -> None:
        predictor = ModelPredictor()

        with self.assertRaisesRegex(ValueError, "Nenhum modelo"):
            predictor.predict(["texto"])

    def test_predict_proba_raises_for_model_without_method(self) -> None:
        model = Mock(spec=["predict"])
        predictor = ModelPredictor(model=model)

        with self.assertRaisesRegex(AttributeError, "predict_proba"):
            predictor.predict_proba(["texto"])

    @patch("ml_prep_kit.model_predictor.mlflow.sklearn.load_model")
    @patch("ml_prep_kit.model_predictor.mlflow.set_tracking_uri")
    def test_from_mlflow_loads_sklearn_model(
        self,
        set_tracking_uri_mock,
        load_model_mock,
    ) -> None:
        loaded_model = Mock()
        load_model_mock.return_value = loaded_model

        predictor = ModelPredictor.from_mlflow(
            model_uri="models:/hospital_triage_text_classifier@champion",
            flavor="sklearn",
            tracking_uri="sqlite:///mlflow.db",
        )

        set_tracking_uri_mock.assert_called_once_with("sqlite:///mlflow.db")
        load_model_mock.assert_called_once_with(
            "models:/hospital_triage_text_classifier@champion"
        )
        self.assertIs(predictor.model, loaded_model)

    @patch("ml_prep_kit.model_predictor.joblib.load")
    def test_load_preprocessor_uses_joblib(self, joblib_load_mock) -> None:
        loaded_preprocessor = Mock()
        joblib_load_mock.return_value = loaded_preprocessor
        predictor = ModelPredictor(model=Mock())

        predictor.load_preprocessor("model/preprocessor.joblib")

        joblib_load_mock.assert_called_once_with("model/preprocessor.joblib")
        self.assertIs(predictor.preprocessor, loaded_preprocessor)


if __name__ == "__main__":
    unittest.main()
