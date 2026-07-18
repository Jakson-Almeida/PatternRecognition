"""Regenera a figura da ablação de features (potências vs +posições Bragg).

Lê results/passo5_compare_features.csv e exclui o AdaBoost, cuja linha no CSV
foi gerada com a configuração antiga (stumps, acc ~0.61) e não reflete o modelo
reajustado reportado no artigo. Salva em classification/figures e ieee/figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
IEEE_FIGURES_DIR = ROOT.parent / "ieee" / "figures"

LABELS = {
    "kNN": "kNN",
    "SVM": "SVM",
    "MLP": "MLP",
    "RandomForest": "Random Forest",
    "MQ": "MQ",
}


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "passo5_compare_features.csv")
    df = df[df["classifier"] != "AdaBoost"].reset_index(drop=True)
    order = ["SVM", "RandomForest", "MLP", "kNN", "MQ"]
    df["classifier"] = pd.Categorical(df["classifier"], categories=order, ordered=True)
    df = df.sort_values("classifier").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    xpos = np.arange(len(df))
    w = 0.35
    ax.bar(xpos - w / 2, df["powers_only"], w, label="só potências", color="#4c72b0")
    ax.bar(xpos + w / 2, df["powers_plus_wl"], w, label="potências + posições Bragg", color="#55a868")
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[c] for c in df["classifier"]])
    ax.set_ylabel("Acurácia CV")
    ax.set_title("Efeito de incluir as posições Bragg na entrada")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    for x, (p, q) in zip(xpos, zip(df["powers_only"], df["powers_plus_wl"])):
        ax.text(x - w / 2, p + 0.015, f"{p:.3f}".replace(".", ","), ha="center", fontsize=8)
        ax.text(x + w / 2, q + 0.015, f"{q:.3f}".replace(".", ","), ha="center", fontsize=8)
    fig.tight_layout()

    for out_dir in (FIGURES_DIR, IEEE_FIGURES_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "passo5_compare_features.png", dpi=150)
        print(f"salvo: {out_dir / 'passo5_compare_features.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
