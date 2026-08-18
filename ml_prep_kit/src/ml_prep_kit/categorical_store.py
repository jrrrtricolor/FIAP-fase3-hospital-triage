import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalStore(BaseEstimator, TransformerMixin):
    """
    Armazena os valores únicos de colunas categóricas para uso posterior.

    Não usar para colunas numéricas ou binárias.

    Exemplo:
        pipeline = Pipeline([
            ("categorical_store", CategoricalStore()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        pipeline.fit(X_train)

        # Retorna todos os valores distintos encontrados no
        # X_train para cada coluna
        categories = pipeline["categorical_store"].categories["coluna"]
    """

    def __init__(self) -> None:
        super().__init__()

        self.categories = {}
        self.is_fitted_ = False

    def fit(self, X, y=None):
        df = pd.DataFrame(X)

        for column in df.columns:
            self.categories[column] = df[column].unique().tolist()

        self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        return X

    def set_output(self, transform):
        return self

    def __getstate__(self):
        return {
            "categories": self.categories,
            "is_fitted_": self.is_fitted_,
        }

    def __setstate__(self, state):
        self.categories = state["categories"]
        self.is_fitted_ = state["is_fitted_"]

    def __sklearn_clone__(self):
        return super().__sklearn_clone__()
