"""Testes do classificador textual PyTorch com interface Scikit-Learn."""

import unittest

import numpy as np

from ml_prep_kit import SklearnTorchTextClassifier


class SklearnTorchTextClassifierTest(unittest.TestCase):
    """Valida treino, transformação e predição multiclasse."""

    def test_fit_predict_and_predict_proba(self) -> None:
        texts = [
            "exame normal sem alteracoes",
            "resultado dentro da normalidade",
            "achado requer avaliacao medica",
            "alteracao merece atencao",
            "risco critico atendimento imediato",
            "urgente intervencao imediata",
        ]
        targets = [
            "normal",
            "normal",
            "atencao",
            "atencao",
            "urgente",
            "urgente",
        ]
        classifier = SklearnTorchTextClassifier(
            embedding_dim=8,
            epochs=2,
            batch_size=2,
            max_vocab_size=100,
            max_sequence_length=12,
            random_seed=42,
        )

        classifier.fit(texts, targets)
        predictions = classifier.predict(texts[:3])
        probabilities = classifier.predict_proba(texts[:3])
        encoded_texts = classifier.transform_texts(texts[:3])

        self.assertEqual(predictions.shape, (3,))
        self.assertEqual(probabilities.shape, (3, 3))
        self.assertEqual(encoded_texts.shape, (3, 12))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(3),
            rtol=1e-5,
        )
        self.assertEqual(set(classifier.classes_), set(targets))

    def test_fit_rejects_empty_text(self) -> None:
        classifier = SklearnTorchTextClassifier(epochs=1)

        with self.assertRaisesRegex(ValueError, "Textos vazios"):
            classifier.fit(["texto válido", ""], ["normal", "urgente"])


if __name__ == "__main__":
    unittest.main()
