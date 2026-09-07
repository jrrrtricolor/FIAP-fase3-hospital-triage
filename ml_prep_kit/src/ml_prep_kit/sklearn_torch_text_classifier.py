"""Classificador textual PyTorch com interface semelhante ao Scikit-Learn."""

import logging
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_prep_kit.text_multiclass_classifier import TextMulticlassClassifier

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


class SklearnTorchTextClassifier(ClassifierMixin, BaseEstimator):
    """Treina um classificador PyTorch leve diretamente sobre textos.

    O wrapper constrói um vocabulário determinístico, transforma cada texto em
    uma sequência de identificadores e treina uma rede com embeddings e mean
    pooling. A interface oferece ``fit``, ``predict`` e ``predict_proba``.

    Exemplo:
        classificador = SklearnTorchTextClassifier(epochs=5)
        classificador.fit(
            ["exame sem alterações", "achado crítico imediato"],
            ["normal", "urgente"],
        )
        probabilidades = classificador.predict_proba(["achado crítico"])
    """

    PAD_TOKEN = "<PAD>"
    UNKNOWN_TOKEN = "<UNK>"

    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64,
        max_vocab_size: int = 20_000,
        max_sequence_length: int = 256,
        min_token_frequency: int = 1,
        lowercase: bool = True,
        random_seed: int = 42,
    ) -> None:
        """Guarda os hiperparâmetros usados na preparação e no treino."""
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.min_token_frequency = min_token_frequency
        self.lowercase = lowercase
        self.random_seed = random_seed

    def fit(
        self,
        X: Sequence[str],
        y: Sequence[object],
    ) -> "SklearnTorchTextClassifier":
        """Constrói o vocabulário e treina o classificador multiclasse."""
        texts = self._validate_texts(X)
        targets = np.asarray(y)

        if len(texts) != len(targets):
            raise ValueError(
                "X e y devem possuir a mesma quantidade de itens."
            )

        self._validate_hyperparameters()
        self.vocabulary_ = self._build_vocabulary(texts)
        self.label_encoder_ = LabelEncoder()
        encoded_targets = self.label_encoder_.fit_transform(targets)
        self.classes_ = self.label_encoder_.classes_

        if len(self.classes_) < 2:
            raise ValueError("O treino exige pelo menos duas classes.")

        torch.manual_seed(self.random_seed)
        encoded_texts = self._encode_texts(texts)
        self.model_module_ = TextMulticlassClassifier(
            vocab_size=len(self.vocabulary_),
            num_classes=len(self.classes_),
            embedding_dim=self.embedding_dim,
            padding_idx=self.vocabulary_[self.PAD_TOKEN],
        )

        training_dataset = TensorDataset(
            torch.tensor(encoded_texts, dtype=torch.long),
            torch.tensor(encoded_targets, dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(self.random_seed)
        training_loader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_module_.parameters(),
            lr=self.learning_rate,
        )

        logger.info(
            "Iniciando o treino do classificador textual PyTorch.",
            extra={
                "evento": "treino_classificador_textual_iniciado",
                "linhas": len(texts),
                "classes": len(self.classes_),
                "vocabulario": len(self.vocabulary_),
                "epocas": self.epochs,
            },
        )

        self.train_loss_ = 0.0
        for epoch in range(1, self.epochs + 1):
            self.train_loss_ = self._train_one_epoch(
                training_loader=training_loader,
                loss_function=loss_function,
                optimizer=optimizer,
            )

            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                logger.info(
                    "Treino textual PyTorch em andamento: época %s/%s.",
                    epoch,
                    self.epochs,
                    extra={
                        "evento": "epoca_classificador_textual_concluida",
                        "epoca": epoch,
                        "total_epocas": self.epochs,
                        "perda_treino": round(self.train_loss_, 6),
                    },
                )

        self.model_module_.eval()
        self.is_fitted_ = True

        logger.info(
            "Treino do classificador textual PyTorch finalizado.",
            extra={
                "evento": "treino_classificador_textual_concluido",
                "perda_treino_final": round(self.train_loss_, 6),
            },
        )

        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Retorna os rótulos originais das classes previstas."""
        predicted_indices = self.predict_proba(X).argmax(axis=1)
        return self.label_encoder_.inverse_transform(predicted_indices)

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Retorna uma probabilidade para cada classe aprendida."""
        check_is_fitted(
            self,
            attributes=[
                "is_fitted_",
                "model_module_",
                "vocabulary_",
                "label_encoder_",
            ],
        )
        texts = self._validate_texts(X)
        encoded_texts = self._encode_texts(texts)
        token_tensor = torch.tensor(encoded_texts, dtype=torch.long)

        self.model_module_.eval()
        with torch.no_grad():
            logits = self.model_module_(token_tensor)
            return torch.softmax(logits, dim=1).numpy()

    def transform_texts(self, X: Sequence[str]) -> np.ndarray:
        """Expõe a tokenização ajustada para exportação e diagnóstico."""
        check_is_fitted(self, attributes=["vocabulary_"])
        texts = self._validate_texts(X)
        return self._encode_texts(texts)

    def _build_vocabulary(self, texts: list[str]) -> dict[str, int]:
        """Cria um vocabulário estável limitado pelos hiperparâmetros."""
        token_counts = Counter(
            token for text in texts for token in self._tokenize(text)
        )
        eligible_tokens = [
            token
            for token, count in token_counts.items()
            if count >= self.min_token_frequency
        ]
        eligible_tokens.sort(key=lambda token: (-token_counts[token], token))

        vocabulary = {
            self.PAD_TOKEN: 0,
            self.UNKNOWN_TOKEN: 1,
        }
        available_slots = self.max_vocab_size - len(vocabulary)

        for token in eligible_tokens[:available_slots]:
            vocabulary[token] = len(vocabulary)

        return vocabulary

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Transforma textos em uma matriz de identificadores de tokens."""
        pad_id = self.vocabulary_[self.PAD_TOKEN]
        unknown_id = self.vocabulary_[self.UNKNOWN_TOKEN]
        encoded = np.full(
            (len(texts), self.max_sequence_length),
            fill_value=pad_id,
            dtype=np.int64,
        )

        for row_index, text in enumerate(texts):
            token_ids = [
                self.vocabulary_.get(token, unknown_id)
                for token in self._tokenize(text)[: self.max_sequence_length]
            ]
            encoded[row_index, : len(token_ids)] = token_ids

        return encoded

    def _tokenize(self, text: str) -> list[str]:
        """Aplica uma tokenização simples e reproduzível."""
        normalized_text = text.lower() if self.lowercase else text
        return TOKEN_PATTERN.findall(normalized_text)

    def _validate_texts(self, X: Sequence[str]) -> list[str]:
        """Valida e materializa a coleção de textos recebida."""
        if isinstance(X, str):
            raise TypeError(
                "X deve ser uma coleção de textos, não uma string."
            )

        texts = list(X)
        if not texts:
            raise ValueError("X deve possuir pelo menos um texto.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Todos os itens de X devem ser textos.")

        if any(not text.strip() for text in texts):
            raise ValueError("Textos vazios não são permitidos.")

        return texts

    def _validate_hyperparameters(self) -> None:
        """Impede configurações que inviabilizam tokenização ou treino."""
        if self.max_vocab_size < 2:
            raise ValueError("max_vocab_size deve ser maior ou igual a 2.")

        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length deve ser positivo.")

        if self.min_token_frequency < 1:
            raise ValueError("min_token_frequency deve ser positivo.")

        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs e batch_size devem ser positivos.")

    def _train_one_epoch(
        self,
        training_loader: DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Executa uma época e retorna a perda média ponderada."""
        self.model_module_.train()
        total_loss = 0.0
        total_rows = 0

        for token_batch, target_batch in training_loader:
            optimizer.zero_grad()
            logits = self.model_module_(token_batch)
            loss = loss_function(logits, target_batch)
            loss.backward()
            optimizer.step()

            batch_size = len(target_batch)
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size

        return total_loss / total_rows
