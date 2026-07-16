"""Utilitários para o experimento de classificação de FBGs."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

RANDOM_STATE = 42
K_DEFAULT = 4
N_FBGS = 13
N_CLASSES = N_FBGS - K_DEFAULT + 1  # 10 janelas contíguas
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


def filter_wl_range(
    X: np.ndarray,
    wl_bragg: np.ndarray,
    target: np.ndarray,
    wl_range: tuple[float, float] = WL_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mantém amostras com wl_range[0] < lambda_res < wl_range[1] (como no notebook 4)."""
    target = np.asarray(target, dtype=float).ravel()
    lo, hi = wl_range
    keep = (target > lo) & (target < hi)
    return (
        np.asarray(X, dtype=float)[keep],
        np.asarray(wl_bragg, dtype=float)[keep],
        target[keep],
        keep,
    )


def make_topk_mask(wl_bragg: np.ndarray, target: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """Máscara: 1 nos k FBGs mais próximos de lambda_res (intermediário do rótulo)."""
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


def class_to_mask(y_class: np.ndarray | int, n_fbgs: int = N_FBGS, k: int = K_DEFAULT) -> np.ndarray:
    """Converte classe(s) de janela s -> máscara com uns em {s,...,s+k-1}."""
    y_class = np.asarray(y_class, dtype=int).ravel()
    n = len(y_class)
    mask = np.zeros((n, n_fbgs), dtype=np.int8)
    for i, s in enumerate(y_class):
        if s < 0 or s > n_fbgs - k:
            raise ValueError(f"classe fora de [0, {n_fbgs - k}]: {s}")
        mask[i, s : s + k] = 1
    return mask


def mask_to_window_class(y_mask: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """
    Converte máscara top-k em classe multiclasse (índice de início da janela).

    Exige janela contígua de exatamente k uns. Classes: C0={0..k-1}, ..., C9={9..12} para k=4.
    """
    y_mask = np.asarray(y_mask, dtype=int)
    if y_mask.ndim != 2:
        raise ValueError("y_mask deve ter shape (n, n_fbgs)")
    n, n_fbgs = y_mask.shape
    max_start = n_fbgs - k
    classes = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx = np.flatnonzero(y_mask[i])
        if len(idx) != k:
            raise ValueError(f"amostra {i}: esperados {k} uns, obtidos {len(idx)}")
        if idx[-1] - idx[0] != k - 1 or not np.all(np.diff(idx) == 1):
            raise ValueError(f"amostra {i}: máscara não contígua {idx.tolist()}")
        s = int(idx[0])
        if s < 0 or s > max_start:
            raise ValueError(f"amostra {i}: início {s} fora de [0, {max_start}]")
        classes[i] = s
    return classes


def prepare_measured_classification(
    k: int = K_DEFAULT,
    wl_range: tuple[float, float] = WL_RANGE,
    path: Path | None = None,
) -> dict:
    """
    Pipeline Passo 1: carrega measured.dataset, normaliza, filtra e gera rótulos.

    Rótulo oficial: y_class (multiclasse, janela contígua de k FBGs).
    y_mask permanece como representação binária equivalente.
    """
    raw = load_measured_dataset(path)
    X_raw = np.asarray(raw["input_strength"], dtype=float)
    wl_bragg_raw = np.asarray(raw["wl_bragg"], dtype=float)
    target_raw = np.asarray(raw["target"], dtype=float).ravel()

    X_norm = normalize_input_strength(X_raw)
    X, wl_bragg, target, keep = filter_wl_range(X_norm, wl_bragg_raw, target_raw, wl_range)
    y_mask = make_topk_mask(wl_bragg, target, k=k)
    y_class = mask_to_window_class(y_mask, k=k)
    # bijeção máscara <-> classe
    assert np.array_equal(class_to_mask(y_class, n_fbgs=X.shape[1], k=k), y_mask)
    assert int(y_class.min()) >= 0 and int(y_class.max()) <= X.shape[1] - k
    assert len(np.unique(y_class)) == (X.shape[1] - k + 1)

    return {
        "X": X,
        "y_class": y_class,
        "y_mask": y_mask,
        "wl_bragg": wl_bragg,
        "target": target,
        "keep": keep,
        "X_raw_shape": np.array(X_raw.shape, dtype=int),
        "n_raw": np.int64(X_raw.shape[0]),
        "n_kept": np.int64(X.shape[0]),
        "k": np.int64(k),
        "n_classes": np.int64(X.shape[1] - k + 1),
        "wl_range": np.array(wl_range, dtype=float),
        "random_state": np.int64(RANDOM_STATE),
    }


def save_prepared_dataset(data: dict, out_path: Path | None = None) -> Path:
    """Salva artefato do Passo 1 em .npz."""
    out_path = out_path or (RESULTS_DIR / "prepared_measured_k4.npz")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return out_path


def load_prepared_dataset(path: Path | None = None) -> dict:
    """Carrega artefato .npz do Passo 1."""
    path = path or (RESULTS_DIR / "prepared_measured_k4.npz")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}