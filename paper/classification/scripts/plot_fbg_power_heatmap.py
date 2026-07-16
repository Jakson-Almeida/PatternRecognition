"""Heatmap 73x13: potências normalizadas de FBGs (subamostra das 7300)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import FIGURES_DIR, load_prepared_dataset

N_SHOW = 73
N_FBGS = 13


def select_samples(X: np.ndarray, target: np.ndarray, n: int = N_SHOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Escolhe n amostras espalhadas ao longo de lambda_res (ordem crescente).

    Assim o heatmap mostra o padrão de potência deslocando-se pelas FBGs.
    """
    order = np.argsort(target)
    idx = np.linspace(0, len(order) - 1, n, dtype=int)
    chosen = order[idx]
    return X[chosen], target[chosen], chosen


def main() -> None:
    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    target = np.asarray(data["target"], dtype=float).ravel()
    assert X.shape[1] == N_FBGS

    X_show, _, _ = select_samples(X, target, N_SHOW)

    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    im = ax.imshow(
        X_show,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Potência normalizada", fontsize=11)

    ax.set_xlabel("Índice da FBG", fontsize=11)
    ax.set_title(f"Potências FBG — {N_SHOW} de {X.shape[0]} amostras", fontsize=12)
    ax.set_xticks(np.arange(N_FBGS))
    ax.set_xticklabels([str(i) for i in range(N_FBGS)])
    ax.set_yticks([])
    ax.set_ylabel("")

    fig.tight_layout()
    out = FIGURES_DIR / "fig_fbg_power_heatmap_73x13.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out}")
    print(f"shape: {X_show.shape}")
    print(f"X min/max na figura: {X_show.min():.4f} / {X_show.max():.4f}")


if __name__ == "__main__":
    main()
