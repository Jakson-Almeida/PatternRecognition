"""Gráfico simples: lambda_res ao longo das 73 amostras (mesma seleção do heatmap)."""

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


def select_lambda_res(target: np.ndarray, n: int = N_SHOW) -> np.ndarray:
    """Mesma subamostra do heatmap: n pontos igualmente espaçados em lambda_res ordenado."""
    order = np.argsort(target)
    idx = np.linspace(0, len(order) - 1, n, dtype=int)
    return target[order[idx]]


def main() -> None:
    data = load_prepared_dataset()
    target = np.asarray(data["target"], dtype=float).ravel()
    lam = select_lambda_res(target, N_SHOW)
    x = np.arange(1, N_SHOW + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, lam, "o-", color="#4c72b0", markersize=4, linewidth=1.2)
    ax.set_xlabel("Amostra (1–73)", fontsize=11)
    ax.set_ylabel(r"$\lambda_{res}$ (nm)", fontsize=11)
    ax.set_title(
        rf"$\lambda_{{res}}$ nas {N_SHOW} amostras selecionadas"
        f"\n({lam[0]:.1f} → {lam[-1]:.1f} nm)",
        fontsize=12,
    )
    ax.set_xlim(0.5, N_SHOW + 0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig_lambda_res_73_samples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out}")
    print(f"n={len(lam)} | [{lam.min():.2f}, {lam.max():.2f}] nm")


if __name__ == "__main__":
    main()
