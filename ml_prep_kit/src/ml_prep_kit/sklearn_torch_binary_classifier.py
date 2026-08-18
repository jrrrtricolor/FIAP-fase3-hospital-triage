"""Classificador PyTorch com interface parecida com Scikit-Learn."""

import logging

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_prep_kit.tabular_binary_classifier import TabularBinaryClassifier

logger = logging.getLogger(__name__)


class SklearnTorchBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Treina uma rede neural PyTorch como classificador binário.

    Use esta classe quando quiser treinar uma rede neural simples com uma
    interface familiar ao Scikit-Learn. Ela recebe dados tabulares já
    transformados em números e expõe os métodos ``fit``, ``predict`` e
    ``predict_proba``.

    Exemplo:
        classificador = SklearnTorchBinaryClassifier(
            hidden_size=32,
            learning_rate=0.001,
            epochs=10,
            batch_size=512,
            random_seed=42,
        )

        classificador.fit(X_treino, y_treino)
        probabilidades = classificador.predict_proba(X_validacao)
    """

    def __init__(
        self,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 512,
        random_seed: int = 42,
        threshold: float = 0.5,
        model_module: nn.Module | None = None,
    ) -> None:
        """Guarda os hiperparâmetros usados no treino.

        Exemplo:
            classificador = SklearnTorchBinaryClassifier(
                hidden_size=64,
                learning_rate=0.001,
                epochs=10,
            )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.threshold = threshold
        self.model_module = model_module
        self.is_fitted_ = False

    def fit(self, X, y):
        """Treina a rede neural com os dados informados.

        Exemplo:
            classificador.fit(X_treino, y_treino)
        """

        if self.is_fitted_:
            raise ValueError("O modelo já foi treinado ou carregado.")

        X_matriz = np.asarray(X, dtype=np.float32)
        y_vetor = np.asarray(y, dtype=np.float32)

        torch.manual_seed(self.random_seed)

        # Quantidade de features usadas como entrada da rede neural.
        quantidade_features = X_matriz.shape[1]
        self.model_module = TabularBinaryClassifier(
            input_size=quantidade_features,
            hidden_size=self.hidden_size,
        )

        # TensorDataset junta features e alvo para o DataLoader.
        conjunto_treino = TensorDataset(
            torch.tensor(X_matriz, dtype=torch.float32),
            torch.tensor(y_vetor, dtype=torch.float32),
        )
        gerador = torch.Generator().manual_seed(self.random_seed)
        carregador_treino = DataLoader(
            conjunto_treino,
            batch_size=self.batch_size,
            shuffle=True,
            generator=gerador,
        )

        # BCEWithLogitsLoss espera a saída bruta da rede, sem sigmoid.
        funcao_perda = nn.BCEWithLogitsLoss()
        otimizador = torch.optim.Adam(
            self.model_module.parameters(),
            lr=self.learning_rate,
        )

        logger.info(
            "Iniciando o ajuste do classificador PyTorch.",
            extra={
                "evento": "treino_classificador_pytorch_iniciado",
                "linhas": len(X_matriz),
                "features": quantidade_features,
                "epocas": self.epochs,
                "tamanho_lote": self.batch_size,
            },
        )

        self.train_loss_ = 0.0
        for epoca in range(1, self.epochs + 1):
            self.train_loss_ = self._train_one_epoch(
                carregador_treino=carregador_treino,
                funcao_perda=funcao_perda,
                otimizador=otimizador,
            )
            if epoca == 1 or epoca % 5 == 0 or epoca == self.epochs:
                logger.info(
                    "Treino PyTorch em andamento: época %s/%s.",
                    epoca,
                    self.epochs,
                    extra={
                        "evento": "epoca_classificador_pytorch_concluida",
                        "epoca": epoca,
                        "total_epocas": self.epochs,
                        "perda_treino": round(self.train_loss_, 6),
                    },
                )

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = quantidade_features
        self.is_fitted_ = True
        self.model_module.eval()

        logger.info(
            "Ajuste do classificador PyTorch finalizado.",
            extra={
                "evento": "treino_classificador_pytorch_concluido",
                "perda_treino_final": round(self.train_loss_, 6),
            },
        )

        return self

    def predict(self, X) -> np.ndarray:
        """Prediz a classe final usando o limiar configurado.

        Exemplo:
            classes = classificador.predict(X_validacao)
        """
        probabilidades = self.predict_proba(X)[:, 1]
        return (probabilidades >= self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Retorna probabilidades das classes negativa e positiva.

        Exemplo:
            probabilidades = classificador.predict_proba(X_validacao)
        """
        if self.is_fitted_ is False:
            if self.model_module is None:
                raise ValueError("Modelo não treinado.")

        X_matriz = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X_matriz, dtype=torch.float32)

        if self.is_fitted_:
            self.model_module.eval()

        with torch.no_grad():
            # Sigmoid transforma a saída bruta em probabilidade positiva.
            logits = self.model_module(X_tensor)
            probabilidade_positiva = torch.sigmoid(logits).numpy()

        probabilidade_negativa = 1 - probabilidade_positiva
        return np.stack(
            (probabilidade_negativa, probabilidade_positiva),
            axis=-1,
        )

    def _train_one_epoch(
        self,
        carregador_treino: DataLoader,
        funcao_perda: nn.Module,
        otimizador: torch.optim.Optimizer,
    ) -> float:
        """Executa uma época de treino.

        Exemplo:
            perda = self._train_one_epoch(
                carregador_treino,
                funcao_perda,
                otimizador,
            )
        """
        self.model_module.train()
        perda_total = 0.0
        total_linhas = 0

        for X_lote, y_lote in carregador_treino:
            # Zera gradientes antigos antes do novo ajuste.
            otimizador.zero_grad()
            logits = self.model_module(X_lote)
            perda = funcao_perda(logits, y_lote)

            # Calcula gradientes e atualiza os pesos da rede.
            perda.backward()
            otimizador.step()

            tamanho_lote = len(y_lote)
            perda_total += float(perda.item()) * tamanho_lote
            total_linhas += tamanho_lote

        return perda_total / total_linhas
