"""Passo 6: figuras IEEE-ready para formulação multiclasse."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import build_classifiers
from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, K_DEFAULT, N_CLASSES, class_to_mask, load_prepared_dataset
from src.tuned_models import build_tuned_classifiers


def fig_schema(out: Path):
    data = load_prepared_dataset()
    wl_mean = np.asarray(data["wl_bragg"], dtype=float).mean(axis=0)
    lam = float(np.median(data["target"]))
    # classe típica perto da mediana
    y = np.asarray(data["y_class"], dtype=int).ravel()
    # pick a sample near median lambda
    target = np.asarray(data["target"], dtype=float).ravel()
    i = int(np.argmin(np.abs(target - lam)))
    s = int(y[i])
    active = set(range(s, s + K_DEFAULT))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axvline(target[i], color="#c44e52", ls="--", lw=1.5, label=r"$\lambda_{res}$")
    for j, w in enumerate(wl_mean):
        color = "#4c72b0" if j in active else "#bbbbbb"
        lw = 2.2 if j in active else 1.0
        ax.plot([w, w], [0, 1], color=color, lw=lw)
        ax.text(w, 1.05, str(j), ha="center", fontsize=8, color=color)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_yticks([])
    ax.set_title(f"Esquema: LPFG + 13 FBGs — classe C{s} = janela {{{','.join(map(str, range(s, s+4)))}}}")
    ax.plot([], [], color="#4c72b0", lw=2.2, label="FBG na janela")
    ax.plot([], [], color="#bbbbbb", lw=1.0, label="FBG fora")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_example(out: Path):
    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()
    clf = build_tuned_classifiers()["SVM"]
    clf.fit(X[dev], y[dev])
    pred_ho = np.asarray(clf.predict(X[ho]), dtype=int).ravel()

    ok = np.where(pred_ho == y[ho])[0]
    bad = np.where(pred_ho != y[ho])[0]
    local = [ok[0]] if len(ok) else [0]
    if len(bad):
        local.append(bad[0])

    fig, axes = plt.subplots(1, len(local), figsize=(4.2 * len(local), 3.4), squeeze=False)
    for ax, li in zip(axes[0], local):
        i = int(ho[li])
        true_s = int(y[i])
        pred_s = int(pred_ho[li])
        ax.stem(np.arange(13), X[i], linefmt="C0-", markerfmt="C0o", basefmt="k-")
        for j in range(true_s, true_s + 4):
            ax.plot(j, X[i, j], "v", color="#c44e52", markersize=10, label="verdadeiro" if j == true_s else None)
        for j in range(pred_s, pred_s + 4):
            ax.plot(j, X[i, j] + 0.02, "^", color="#55a868", markersize=9, label="previsto" if j == pred_s else None)
        ax.set_xlabel("FBG")
        ax.set_ylabel("Potência norm.")
        ax.set_title(f"true C{true_s} / pred C{pred_s}")
        ax.legend(fontsize=7)
    fig.suptitle("Janela verdadeira (▼) vs SVM (▲)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_schema(FIGURES_DIR / "fig_schema_lpg_fbg.png")
    fig_example(FIGURES_DIR / "fig_mask_true_vs_pred.png")
    # copy metrics already made in passo5
    print("DONE figures")


if __name__ == "__main__":
    main()
