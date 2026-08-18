"""Rede neural leve para classificação multiclasse de textos tokenizados."""

import torch
from torch import nn


class TextMulticlassClassifier(nn.Module):
    """Classifica sequências de tokens com embeddings e mean pooling.

    A rede recebe uma matriz de identificadores de tokens no formato
    ``(linhas, comprimento_sequencia)``. Os embeddings dos tokens válidos são
    agregados pela média e projetados para um logit por classe.

    Exemplo:
        modelo = TextMulticlassClassifier(
            vocab_size=100,
            num_classes=3,
            embedding_dim=32,
        )
        logits = modelo(torch.ones((4, 20), dtype=torch.long))
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        padding_idx: int = 0,
    ) -> None:
        """Define a camada de embeddings e a projeção multiclasse."""
        super().__init__()

        if vocab_size < 2:
            raise ValueError("vocab_size deve possuir pelo menos 2 tokens.")

        if num_classes < 2:
            raise ValueError("num_classes deve possuir pelo menos 2 classes.")

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.output_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Retorna um logit por classe para cada texto tokenizado."""
        embeddings = self.embedding(token_ids)
        valid_token_mask = token_ids.ne(self.padding_idx).unsqueeze(dim=-1)

        embedding_sum = (embeddings * valid_token_mask).sum(dim=1)
        valid_token_count = valid_token_mask.sum(dim=1).clamp_min(1)
        pooled_embeddings = embedding_sum / valid_token_count

        return self.output_layer(pooled_embeddings)
