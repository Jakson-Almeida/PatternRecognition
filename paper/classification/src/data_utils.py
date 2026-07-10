"""Utilitários para o experimento de classificação de FBGs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

RANDOM_STATE = 42
K_DEFAULT = 4
WL_RANGE = (1515.0, 1585.0)

# paper/classification/src -> classification/ -> paper/
CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = CLASSIFICATION_ROOT.parent
REPO_DATA = PAPER_ROOT / "fbg-demodulated-lpfg" / "data"
FIGURES_DIR = CLASSIFICATION_ROOT / "figures"
RESULTS_DIR = CLASSIFICATION_ROOT / "results"


def load_measured_dataset(path: Path | None = None) -> dict:
    """Carrega measured.dataset (dict com input_strength, wl_bragg, target)."""
    path = path or (REPO_DATA / "measured.dataset")
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = {"input_strength", "wl_bragg", "target"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Campos ausentes em {path}: {missing}")
    return data


def normalize_input_strength(X: np.ndarray) -> np.ndarray:
    """Normalização do notebook 4 do Barino: subtrai min e divide pela soma."""
    X = np.asarray(X, dtype=float).copy()
    X = X - X.min(axis=1, keepdims=True)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return X / row_sum


def make_topk_mask(wl_bragg: np.ndarray, target: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """Máscara multi-rótulo: 1 nos k FBGs mais próximos de lambda_res."""
    wl_bragg = np.asarray(wl_bragg, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    if wl_bragg.ndim != 2:
        raise ValueError("wl_bragg deve ter shape (n, n_fbgs)")
    n, n_fbgs = wl_bragg.shape
    if k < 1 or k > n_fbgs:
        raise ValueError(f"k deve estar em [1, {n_fbgs}], recebido {k}")
    err = np.abs(wl_bragg - target.reshape(-1, 1))
    # índices dos k menores erros por linha
    topk_idx = np.argpartition(err, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n_fbgs), dtype=np.int8)
    rows = np.arange(n).reshape(-1, 1)
    mask[rows, topk_idx] = 1
    return mask


def filter_wl_range(
    X: np.ndarray,
    wl_bragg: np.ndarray,
    y: np.ndarray,
    wl_range: tuple[float, float] = WL_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mantém amostras com target em (low, high)."""
    low, high = wl_range
    keep = (y > low) & (y < high)
    return X[keep], wl_bragg[keep], y[keep], keep
