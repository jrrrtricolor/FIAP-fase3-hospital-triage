import unittest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml_prep_kit import ModelFactory


class TestModelFactory(unittest.TestCase):
    def test_list_available_models_returns_model_groups(self):
        factory = ModelFactory()

        available_models = factory.list_available_models()

        self.assertIn("classification", available_models)
        self.assertIn("regression", available_models)
        self.assertIn("random_forest", available_models["classification"])
        self.assertIn("linear_regression", available_models["regression"])

    def test_create_returns_configured_model(self):
        factory = ModelFactory(random_state=42)

        model = factory.create(
            problem_type="classification",
            model_name="random-forest",
            parameters={"n_estimators": 10},
        )

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.random_state, 42)

    def test_create_pipeline_returns_sklearn_pipeline(self):
        factory = ModelFactory(random_state=42)
        preprocessor = ColumnTransformer(
            transformers=[("keep", "passthrough", ["feature"])]
        )

        pipeline = factory.create_pipeline(
            preprocessor=preprocessor,
            problem_type="classification",
            model_name="logistic_regression",
            parameters={"max_iter": 1000},
        )

        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(
            list(pipeline.named_steps),
            ["preprocessor", "model"],
        )

    def test_create_raises_error_for_unknown_model(self):
        factory = ModelFactory()

        with self.assertRaisesRegex(ValueError, "não disponível"):
            factory.create(
                problem_type="classification",
                model_name="unknown_model",
            )


if __name__ == "__main__":
    unittest.main()
