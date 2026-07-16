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

    X_show, lam_show, _ = select_samples(X, target, N_SHOW)

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
    ax.set_ylabel("Amostra (ordenada por $\\lambda_{res}$ crescente)", fontsize=11)
    ax.set_title(
        f"Potências FBG — {N_SHOW} de {X.shape[0]} amostras\n"
        f"$\\lambda_{{res}}$: {lam_show[0]:.1f} → {lam_show[-1]:.1f} nm",
        fontsize=12,
    )
    ax.set_xticks(np.arange(N_FBGS))
    ax.set_xticklabels([str(i) for i in range(N_FBGS)])

    # Marcas de lambda_res no eixo y (algumas linhas)
    y_ticks = np.linspace(0, N_SHOW - 1, 8, dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{lam_show[i]:.0f}" for i in y_ticks])
    ax.set_ylabel("$\\lambda_{res}$ da amostra (nm)", fontsize=11)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_fbg_power_heatmap_73x13.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out}")
    print(f"shape: {X_show.shape} | lambda [{lam_show.min():.2f}, {lam_show.max():.2f}] nm")
    print(f"X min/max na figura: {X_show.min():.4f} / {X_show.max():.4f}")


if __name__ == "__main__":
    main()
