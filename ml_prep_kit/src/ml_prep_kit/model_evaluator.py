"""Métricas reutilizáveis para avaliação de classificadores."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Calcula métricas consistentes para classificação binária ou multiclasse."""

    def evaluate_classification(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
        y_score: pd.Series | pd.DataFrame | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Retorna métricas globais de classificação.

        As métricas macro e weighted estão sempre presentes. Para manter
        compatibilidade com fluxos binários, ``precision``, ``recall`` e ``f1``
        representam a classe positiva em problemas 0/1 e a média macro em
        problemas multiclasse.
        """
        labels = np.unique(np.asarray(y_true))
        is_binary = len(labels) == 2
        default_average = "binary" if is_binary else "macro"
        average_options = {
            "average": default_average,
            "zero_division": 0,
        }
        if is_binary:
            average_options["pos_label"] = labels[-1]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                **average_options,
            ),
            "precision_macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "precision_weighted": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        if y_score is not None and len(labels) > 1:
            score_array = np.asarray(y_score)

            if is_binary:
                if score_array.ndim == 2:
                    score_array = score_array[:, -1]
                metrics["roc_auc"] = roc_auc_score(y_true, score_array)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true,
                    score_array,
                    labels=labels,
                    multi_class="ovr",
                    average="macro",
                )

        return {name: float(value) for name, value in metrics.items()}

    def evaluate_classification_by_class(
        self,
        y_true: pd.Series | np.ndarray | list,
        y_pred: pd.Series | np.ndarray | list,
    ) -> pd.DataFrame:
        """Retorna precision, recall, F1 e suporte para cada classe."""
        labels = np.unique(np.asarray(y_true))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        return pd.DataFrame(
            {
                "class": labels,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
