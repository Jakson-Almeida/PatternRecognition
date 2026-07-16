"""Métricas multiclasse (Passo 3+)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .data_utils import N_CLASSES


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Métricas principais alinhadas à disciplina."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    labels = np.arange(N_CLASSES) if labels is None else np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        ),
    }


def multiclass_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    labels = np.arange(N_CLASSES) if labels is None else np.asarray(labels)
    return confusion_matrix(
        np.asarray(y_true).ravel(),
        np.asarray(y_pred).ravel(),
        labels=labels,
        normalize=normalize,
    )


# Compatibilidade: alguns scripts antigos importavam estes nomes
def evaluate_multilabel(y_true, y_pred):  # pragma: no cover
    raise RuntimeError("Formulação multi-rótulo removida; use evaluate_multiclass.")


def topk_from_scores(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("top-k removido; predição multiclasse via predict().")


def extract_scores(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("extract_scores removido; use predict() multiclasse.")
