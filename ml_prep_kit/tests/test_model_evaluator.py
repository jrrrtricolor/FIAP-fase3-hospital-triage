import unittest

import pandas as pd

from ml_prep_kit import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def test_evaluate_classification_returns_expected_metrics(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([0, 1, 1, 1])
        y_score = pd.Series([0.1, 0.6, 0.8, 0.9])

        metrics = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIn("roc_auc", metrics)

    def test_evaluate_multiclass_returns_macro_and_per_class_metrics(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series(
            ["normal", "normal", "atencao", "atencao", "urgente", "urgente"]
        )
        y_pred = pd.Series(
            ["normal", "atencao", "atencao", "atencao", "urgente", "normal"]
        )

        metrics = evaluator.evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
        )
        class_metrics = evaluator.evaluate_classification_by_class(
            y_true=y_true,
            y_pred=y_pred,
        )

        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertEqual(set(class_metrics["class"]), {"normal", "atencao", "urgente"})
        self.assertEqual(
            set(class_metrics.columns),
            {"class", "precision", "recall", "f1", "support"},
        )


if __name__ == "__main__":
    unittest.main()
