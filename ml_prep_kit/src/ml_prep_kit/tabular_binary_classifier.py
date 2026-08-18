"""Rede neural reutilizável para classificação binária tabular."""

import torch
from torch import nn


class TabularBinaryClassifier(nn.Module):
    """Rede neural para classificação binária com dados tabulares.

    Estrutura da rede:
    entrada -> camada oculta -> ReLU -> camada de saída

    Use esta classe quando já tiver uma matriz numérica de features e precisar
    prever a probabilidade de uma classe positiva. A rede retorna logits, sem
    aplicar sigmoid, pois isso será tratado pela ``BCEWithLogitsLoss`` no
    treino.

    Exemplo:
        modelo = TabularBinaryClassifier(
            input_size=10,
            hidden_size=32,
        )

        logits = modelo(features)
        probabilidades = torch.sigmoid(logits)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
    ) -> None:
        """Define a arquitetura feed-forward usada na classificação.

        Exemplo:
            modelo = TabularBinaryClassifier(
                input_size=X_train.shape[1],
                hidden_size=32,
            )
        """
        super().__init__()
        # Camada que conecta as features de entrada à camada oculta.
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        # Função de ativação que introduz não linearidade no modelo.
        self.activation = nn.ReLU()

        # Camada que conecta a camada oculta à saída.
        # A saída tem 1 neurônio porque o problema é binário.
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Calcula um logit da classe positiva para cada linha.

        Exemplo:
            logits = modelo(torch.ones(3, 10))
        """
        # Passo 1: entrada passa pela camada oculta.
        features = self.hidden_layer(features)

        # Passo 2: aplica função de ativação.
        features = self.activation(features)

        # Passo 3: calcula a saída final como valor bruto da classe positiva.
        features = self.output_layer(features)

        # Não aplicamos sigmoid aqui; a loss transforma esse valor em
        # probabilidade durante o treino.
        return features.squeeze(dim=1)
