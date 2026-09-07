"""Fachada lazy para os componentes reutilizáveis do framework."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CategoricalStore": ("categorical_store", "CategoricalStore"),
    "CSVDataLoader": ("csv_data_loader", "CSVDataLoader"),
    "DataValidator": ("data_validator", "DataValidator"),
    "ExperimentTracker": ("experiment_tracker", "ExperimentTracker"),
    "FeaturePreprocessor": ("feature_preprocessor", "FeaturePreprocessor"),
    "ModelEvaluator": ("model_evaluator", "ModelEvaluator"),
    "ModelFactory": ("model_factory", "ModelFactory"),
    "ModelPredictor": ("model_predictor", "ModelPredictor"),
    "SklearnTorchBinaryClassifier": (
        "sklearn_torch_binary_classifier",
        "SklearnTorchBinaryClassifier",
    ),
    "SklearnTorchTextClassifier": (
        "sklearn_torch_text_classifier",
        "SklearnTorchTextClassifier",
    ),
    "SQLiteDataFrameStore": (
        "sqlite_dataframe_store",
        "SQLiteDataFrameStore",
    ),
    "StructuredJsonFormatter": (
        "structured_json_formatter",
        "StructuredJsonFormatter",
    ),
    "StructuredLoggingConfigurator": (
        "structured_logging_configurator",
        "StructuredLoggingConfigurator",
    ),
    "TabularBinaryClassifier": (
        "tabular_binary_classifier",
        "TabularBinaryClassifier",
    ),
    "TextMulticlassClassifier": (
        "text_multiclass_classifier",
        "TextMulticlassClassifier",
    ),
    "VisualizationReporter": (
        "visualization_reporter",
        "VisualizationReporter",
    ),
    "format_currency": ("utils", "format_currency"),
    "format_percent": ("utils", "format_percent"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Importa somente o componente solicitado pelo consumidor."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(f"ml_prep_kit.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
