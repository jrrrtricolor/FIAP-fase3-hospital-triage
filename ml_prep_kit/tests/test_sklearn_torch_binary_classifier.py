"""Testes do classificador PyTorch com interface Scikit-Learn."""

import logging
import unittest

import numpy as np

from ml_prep_kit import (
    SklearnTorchBinaryClassifier,
    StructuredLoggingConfigurator,
)

logger = logging.getLogger(__name__)


class SklearnTorchBinaryClassifierTest(unittest.TestCase):
    """Valida o treino e a predição do classificador PyTorch.

    Exemplo:
        classificador = SklearnTorchBinaryClassifier(epochs=2)
        classificador.fit(X, y)
    """

    def test_fit_predict_and_predict_proba(self) -> None:
        """O classificador deve treinar e retornar predições válidas.

        Exemplo:
            probabilidades = classificador.predict_proba(X)
            classes = classificador.predict(X)
        """
        StructuredLoggingConfigurator.configure()
        logger.info(
            "Validando classificador PyTorch com interface Scikit-Learn.",
            extra={"evento": "teste_sklearn_torch_iniciado"},
        )

        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1], dtype=np.float32)

        classificador = SklearnTorchBinaryClassifier(
            hidden_size=4,
            epochs=2,
            batch_size=2,
            random_seed=42,
        )
        classificador.fit(X, y)

        classes = classificador.predict(X)
        probabilidades = classificador.predict_proba(X)

        self.assertEqual(classes.shape, (4,))
        self.assertEqual(probabilidades.shape, (4, 2))
        self.assertTrue(hasattr(classificador, "model_module"))


if __name__ == "__main__":
    unittest.main()
