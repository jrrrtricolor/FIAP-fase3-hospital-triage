"""Testes da rede neural textual multiclasse."""

import unittest

import torch

from ml_prep_kit import TextMulticlassClassifier


class TextMulticlassClassifierTest(unittest.TestCase):
    """Valida o contrato de entrada e saída da rede textual."""

    def test_forward_returns_one_logit_per_class(self) -> None:
        model = TextMulticlassClassifier(
            vocab_size=20,
            num_classes=3,
            embedding_dim=8,
        )
        token_ids = torch.tensor(
            [
                [2, 3, 4, 0],
                [5, 6, 0, 0],
            ],
            dtype=torch.long,
        )

        logits = model(token_ids)

        self.assertEqual(logits.shape, torch.Size([2, 3]))

    def test_constructor_rejects_single_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "pelo menos 2 classes"):
            TextMulticlassClassifier(vocab_size=20, num_classes=1)


if __name__ == "__main__":
    unittest.main()
