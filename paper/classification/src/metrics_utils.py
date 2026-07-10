"""Métricas multi-rótulo e predição top-k (Passo 3)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)


def topk_from_scores(scores: np.ndarray, k: int = 4) -> np.ndarray:
    """Converte scores (n, n_fbgs) em máscara com exatamente k uns por linha."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores deve ter shape (n, n_fbgs)")
    n, n_fbgs = scores.shape
    if k < 1 or k > n_fbgs:
        raise ValueError(f"k fora de [1, {n_fbgs}]")
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n_fbgs), dtype=np.int8)
    mask[np.arange(n).reshape(-1, 1), topk_idx] = 1
    return mask


def extract_scores(estimator, X: np.ndarray) -> np.ndarray:
    """
    Scores por FBG para ranquear top-k.

    Ordem de preferência: predict_proba (classe 1) → decision_function → predict.
    """
    X = np.asarray(X, dtype=float)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, list):
            # multilabel nativo (ex.: RF): lista de (n, 2)
            cols = []
            for p in proba:
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] >= 2:
                    cols.append(p[:, 1])
                else:
                    cols.append(p.ravel())
            return np.column_stack(cols)
        proba = np.asarray(proba)
        if proba.ndim == 3:
            # (n_labels, n, n_classes) em alguns wrappers
            return proba[:, :, 1].T if proba.shape[0] < proba.shape[1] else proba[:, :, 1]
        if proba.ndim == 2:
            return proba
        raise TypeError(f"predict_proba com shape inesperado: {proba.shape}")

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim == 1:
            return scores.reshape(-1, 1)
        return scores

    pred = np.asarray(estimator.predict(X), dtype=float)
    if pred.ndim == 1:
        return pred.reshape(-1, 1)
    return pred


def set_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Média, por amostra, de |Y ∩ Ŷ| / |Y| (com |Y|=k no nosso problema)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    inter = (y_true.astype(bool) & y_pred.astype(bool)).sum(axis=1)
    denom = y_true.sum(axis=1).astype(float)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(inter / denom))


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métricas pedidas no guia (média sobre o fold)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "jaccard_samples": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "set_recall": set_recall_at_k(y_true, y_pred),
        "mean_pred_positives": float(y_pred.sum(axis=1).mean()),
    }
