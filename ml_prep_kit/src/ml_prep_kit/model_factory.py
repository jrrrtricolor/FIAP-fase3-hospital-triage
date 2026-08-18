"""Fábrica reutilizável para criação de modelos scikit-learn."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelFactory:
    """Cria modelos e pipelines de machine learning de forma padronizada.

    Use esta classe para evitar que cada notebook instancie modelos de uma
    forma diferente. A fábrica recebe o tipo de problema, o nome do modelo e
    os parâmetros desejados. Ela devolve uma instância pronta para treino.

    Exemplo:
        factory = ModelFactory(random_state=42)

        model = factory.create(
            problem_type="classification",
            model_name="random_forest",
            parameters={"n_estimators": 100},
        )

        pipeline = factory.create_pipeline(
            preprocessor=feature_preprocessor.create_pipeline(),
            problem_type="classification",
            model_name="logistic_regression",
        )
    """

    CLASSIFICATION_MODELS = {
        # Modelo base para comparar se os demais modelos realmente aprendem.
        "dummy": DummyClassifier,
        # Árvore simples, fácil de interpretar e útil como primeiro teste.
        "decision_tree": DecisionTreeClassifier,
        # Combina árvores em sequência para capturar padrões mais complexos.
        "gradient_boosting": GradientBoostingClassifier,
        # Versão otimizada do boosting para bases maiores.
        "hist_gradient_boosting": HistGradientBoostingClassifier,
        # Usa exemplos parecidos para classificar novos registros.
        "knn": KNeighborsClassifier,
        # Modelo linear forte para classificação binária ou multiclasse.
        "logistic_regression": LogisticRegression,
        # Combina várias árvores para reduzir variação e melhorar estabilidade.
        "random_forest": RandomForestClassifier,
    }

    REGRESSION_MODELS = {
        # Árvore simples para prever valores numéricos com boa interpretação.
        "decision_tree": DecisionTreeRegressor,
        # Modelo base para comparar se os demais modelos agregam valor.
        "dummy": DummyRegressor,
        # Combina árvores em sequência para reduzir erro de previsão.
        "gradient_boosting": GradientBoostingRegressor,
        # Versão otimizada do boosting para bases maiores.
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        # Usa exemplos parecidos para estimar valores numéricos.
        "knn": KNeighborsRegressor,
        # Modelo linear simples para relações aproximadamente proporcionais.
        "linear_regression": LinearRegression,
        # Combina várias árvores para previsões mais estáveis.
        "random_forest": RandomForestRegressor,
    }

    RANDOM_STATE_MODELS = {
        "decision_tree",
        "gradient_boosting",
        "hist_gradient_boosting",
        "logistic_regression",
        "random_forest",
    }

    def __init__(self, random_state: int | None = 42) -> None:
        """Define o estado aleatório padrão dos modelos compatíveis.

        O random_state ajuda a reproduzir os mesmos resultados em diferentes
        execuções. Ele só é aplicado em modelos que aceitam esse parâmetro.

        Exemplo:
            factory = ModelFactory(random_state=42)
        """
        self.random_state = random_state

    def list_available_models(
        self,
        problem_type: str | None = None,
    ) -> dict[str, list[str]]:
        """Lista os modelos disponíveis por tipo de problema.

        Quando problem_type não é informado, o método retorna os modelos de
        classificação e regressão. Quando informado, retorna apenas a lista do
        tipo solicitado.

        Exemplo:
            available_models = factory.list_available_models()
            classifiers = factory.list_available_models("classification")
        """
        if problem_type is None:
            return {
                "classification": sorted(self.CLASSIFICATION_MODELS),
                "regression": sorted(self.REGRESSION_MODELS),
            }

        model_registry = self._get_model_registry(problem_type)
        normalized_problem_type = self._normalize_problem_type(problem_type)
        return {normalized_problem_type: sorted(model_registry)}

    def create(
        self,
        problem_type: str,
        model_name: str,
        parameters: dict | None = None,
    ) -> BaseEstimator:
        """Cria uma instância de modelo a partir do nome informado.

        O método valida o tipo de problema, localiza o modelo e aplica os
        parâmetros recebidos. Se o modelo aceitar random_state e o usuário não
        passar outro valor, a fábrica usa o random_state definido na classe.

        Exemplo:
            model = factory.create(
                problem_type="classification",
                model_name="random_forest",
                parameters={"n_estimators": 200, "max_depth": 8},
            )
        """
        normalized_model_name = self._normalize_model_name(model_name)
        model_registry = self._get_model_registry(problem_type)

        if normalized_model_name not in model_registry:
            available_models = ", ".join(sorted(model_registry))
            raise ValueError(
                f"Modelo '{model_name}' não disponível para "
                f"'{problem_type}'. Modelos disponíveis: {available_models}."
            )

        model_parameters = dict(parameters or {})
        if self._should_apply_random_state(normalized_model_name):
            model_parameters.setdefault("random_state", self.random_state)

        model_class = model_registry[normalized_model_name]
        return model_class(**model_parameters)

    def create_pipeline(
        self,
        preprocessor: BaseEstimator,
        problem_type: str,
        model_name: str,
        parameters: dict | None = None,
    ) -> Pipeline:
        """Cria um pipeline com preparação de dados e modelo.

        Use este método quando quiser treinar o preprocessador e o modelo no
        mesmo fluxo do scikit-learn. O primeiro passo prepara as variáveis e o
        segundo passo treina o modelo escolhido.

        Exemplo:
            pipeline = factory.create_pipeline(
                preprocessor=feature_preprocessor.create_pipeline(),
                problem_type="classification",
                model_name="logistic_regression",
                parameters={"max_iter": 1000},
            )

            pipeline.fit(X_train, y_train)
        """
        model = self.create(
            problem_type=problem_type,
            model_name=model_name,
            parameters=parameters,
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def _get_model_registry(
        self,
        problem_type: str,
    ) -> dict[str, type[BaseEstimator]]:
        """Retorna o dicionário de modelos para o tipo de problema."""
        normalized_problem_type = self._normalize_problem_type(problem_type)

        if normalized_problem_type == "classification":
            return self.CLASSIFICATION_MODELS

        if normalized_problem_type == "regression":
            return self.REGRESSION_MODELS

        raise ValueError(
            "Tipo de problema inválido. Use 'classification' ou 'regression'."
        )

    def _normalize_problem_type(self, problem_type: str) -> str:
        """Padroniza o tipo de problema recebido."""
        return problem_type.strip().lower()

    def _normalize_model_name(self, model_name: str) -> str:
        """Padroniza o nome do modelo recebido."""
        return model_name.strip().lower().replace("-", "_").replace(" ", "_")

    def _should_apply_random_state(self, model_name: str) -> bool:
        """Indica se o modelo deve receber random_state automaticamente."""
        return (
            self.random_state is not None
            and model_name in self.RANDOM_STATE_MODELS
        )
