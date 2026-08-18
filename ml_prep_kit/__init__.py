"""Atalho local para o pacote mantido em ``ml_prep_kit/src``.

Este arquivo permite importar ``ml_prep_kit`` ao executar testes e notebooks
diretamente da raiz do projeto, sem depender de ``PYTHONPATH`` manual.
"""

# O ajuste de ``__path__`` precisa ocorrer antes dos imports públicos.
# ruff: noqa: E402

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "ml_prep_kit"
__path__.append(str(_PACKAGE_DIR))

from ml_prep_kit.categorical_store import CategoricalStore
from ml_prep_kit.csv_data_loader import CSVDataLoader
from ml_prep_kit.data_validator import DataValidator
from ml_prep_kit.experiment_tracker import ExperimentTracker
from ml_prep_kit.feature_preprocessor import FeaturePreprocessor
from ml_prep_kit.model_evaluator import ModelEvaluator
from ml_prep_kit.model_factory import ModelFactory
from ml_prep_kit.model_predictor import ModelPredictor
from ml_prep_kit.sklearn_torch_binary_classifier import (
    SklearnTorchBinaryClassifier,
)
from ml_prep_kit.sklearn_torch_text_classifier import (
    SklearnTorchTextClassifier,
)
from ml_prep_kit.sqlite_dataframe_store import SQLiteDataFrameStore
from ml_prep_kit.structured_json_formatter import StructuredJsonFormatter
from ml_prep_kit.structured_logging_configurator import (
    StructuredLoggingConfigurator,
)
from ml_prep_kit.tabular_binary_classifier import TabularBinaryClassifier
from ml_prep_kit.text_multiclass_classifier import TextMulticlassClassifier
from ml_prep_kit.utils import format_currency, format_percent
from ml_prep_kit.visualization_reporter import VisualizationReporter

__all__ = [
    "CategoricalStore",
    "CSVDataLoader",
    "DataValidator",
    "ExperimentTracker",
    "FeaturePreprocessor",
    "ModelEvaluator",
    "ModelFactory",
    "ModelPredictor",
    "SklearnTorchBinaryClassifier",
    "SklearnTorchTextClassifier",
    "SQLiteDataFrameStore",
    "StructuredJsonFormatter",
    "StructuredLoggingConfigurator",
    "TabularBinaryClassifier",
    "TextMulticlassClassifier",
    "VisualizationReporter",
    "format_currency",
    "format_percent",
]
