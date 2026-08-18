"""Utilitários reutilizáveis de pré-processamento com scikit-learn."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .categorical_store import CategoricalStore


class FeaturePreprocessor:
    """Prepara variáveis para modelos de machine learning.

    Use esta classe para aplicar a mesma regra de preparação em treino,
    validação e teste. Colunas binárias são mantidas como estão, colunas
    numéricas são preenchidas e normalizadas com Min-Max Scaling, e colunas
    categóricas são convertidas em colunas binárias.

    Exemplo:
        preprocessor = FeaturePreprocessor(
            numeric_columns=["price"],
            categorical_columns=["department"],
            binary_columns=["is_active"],
            min_max_range=(0, 1),
        )

        X_train_ready = preprocessor.fit_prepare(X_train)
        X_valid_ready = preprocessor.prepare(X_valid)
    """

    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        binary_columns: list[str] | None = None,
        min_max_range: tuple[float, float] = (0, 1),
    ) -> None:
        """Define quais colunas receberão cada tipo de tratamento.

        As colunas devem ser separadas antes da criação do pipeline:
        numéricas são preenchidas e normalizadas, categóricas são
        transformadas em colunas binárias e binárias são mantidas sem
        alteração. O parâmetro min_max_range define o intervalo usado na
        normalização das colunas numéricas.

        Exemplo:
            preprocessor = FeaturePreprocessor(
                numeric_columns=["price", "order_count"],
                categorical_columns=["department"],
                binary_columns=["has_reordered"],
                min_max_range=(0, 1),
            )
        """
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.binary_columns = binary_columns or []
        self.min_max_range = min_max_range
        self.transformer: ColumnTransformer | None = None

    def create_pipeline(self) -> ColumnTransformer:
        """Monta o pipeline de preparação com base nos grupos de colunas.

        O pipeline aplica três regras:
        colunas binárias passam direto, colunas numéricas recebem a mediana
        quando há valores ausentes e são normalizadas com MinMaxScaler, e
        colunas categóricas recebem o valor mais frequente quando há valores
        ausentes e são convertidas com one-hot encoding.

        Exemplo:
            pipeline = preprocessor.create_pipeline()
        """
        transformers: list[tuple[str, object, list[str]]] = []

        if self.binary_columns:
            # Colunas binárias já estão em 0/1, então não precisam de escala.
            transformers.append(("bin", "passthrough", self.binary_columns))

        if self.numeric_columns:
            numeric_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "min_max_scaler",
                        MinMaxScaler(feature_range=self.min_max_range),
                    ),
                ]
            )
            transformers.append(("num", numeric_pipeline, self.numeric_columns))

        if self.categorical_columns:
            categorical_pipeline = Pipeline(
                [
                    ("categorical_store", CategoricalStore()),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )
            transformers.append(("cat", categorical_pipeline, self.categorical_columns))

        self.transformer = ColumnTransformer(transformers)
        self.transformer = self.transformer.set_output(transform="pandas")

        return self.transformer

    def fit_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajusta o pipeline nos dados de treino e prepara as variáveis.

        Use este método somente no conjunto de treino. Ele aprende as regras
        necessárias para preparar os dados, como medianas das colunas
        numéricas, valores mais frequentes das colunas categóricas e categorias
        usadas no one-hot encoding.

        Exemplo:
            X_train_ready = preprocessor.fit_prepare(X_train)
        """
        self.create_pipeline()
        return self.transformer.fit_transform(X)

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica nos novos dados o pipeline ajustado no treino.

        Use este método para validação, teste ou dados novos. Ele reutiliza as
        regras aprendidas em fit_prepare(), garantindo que todas as bases sejam
        preparadas do mesmo jeito.

        Exemplo:
            X_valid_ready = preprocessor.prepare(X_valid)
        """
        if self.transformer is None:
            raise ValueError("Ajuste o pipeline antes de chamar prepare().")

        return self.transformer.transform(X)
