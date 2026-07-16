"""Gera matrizes de confusão (agregada e por FBG) no hold-out."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import predict_topk_mask
from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, K_DEFAULT, RESULTS_DIR, load_prepared_dataset
from src.tuned_models import build_tuned_classifiers


def plot_agg(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay(cm, display_labels=["0 (fora)", "1 (máscara)"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    tn, fp, fn, tp = cm.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def plot_per_fbg(y_true, y_pred, title, out_path):
    n_fbgs = y_true.shape[1]
    ncols, nrows = 5, int(np.ceil(n_fbgs / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4 * nrows))
    axes = np.atleast_2d(axes)
    for j in range(n_fbgs):
        r, c = divmod(j, ncols)
        ax = axes[r, c]
        cm = confusion_matrix(y_true[:, j], y_pred[:, j], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(f"FBG {j}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(n_fbgs, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = FIGURES_DIR / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_mask"], dtype=int)
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()

    X_tr, y_tr = X[dev], y[dev]
    X_te, y_te = X[ho], y[ho]
    print(f"dev={len(dev)} holdout={len(ho)} k={K_DEFAULT}")

    preds = {}
    for name, est in build_tuned_classifiers().items():
        print(f"fit {name}...")
        est.fit(X_tr, y_tr)
        preds[name] = predict_topk_mask(est, X_te, k=K_DEFAULT)

    rows = []
    for name, y_pred in preds.items():
        stats = plot_agg(
            y_te,
            y_pred,
            f"{name} — confusão agregada (hold-out)",
            out_dir / f"cm_agg_{name}.png",
        )
        stats["classifier"] = name
        rows.append(stats)
        plot_per_fbg(
            y_te,
            y_pred,
            f"{name} — confusão por FBG (hold-out)",
            out_dir / f"cm_per_fbg_{name}.png",
        )
        print(name, stats)

    pd.DataFrame(rows)[["classifier", "TN", "FP", "FN", "TP"]].to_csv(
        RESULTS_DIR / "confusion_agg_holdout.csv", index=False
    )

    names = list(preds.keys())
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_te.ravel(), preds[name].ravel(), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
    fig.suptitle("Matrizes agregadas — hold-out (top-4)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cm_agg_all_classifiers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Grade equivalente com percentuais (normalização por linha = classe verdadeira)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        cm = confusion_matrix(
            y_te.ravel(),
            preds[name].ravel(),
            labels=[0, 1],
            normalize="true",
        )
        ConfusionMatrixDisplay(cm * 100.0, display_labels=["0", "1"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".1f"
        )
        # Sufixo % em cada célula de texto do display
        for txt in ax.texts:
            txt.set_text(f"{txt.get_text()}%")
        ax.set_title(name)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
    fig.suptitle(
        "Matrizes agregadas (%) — hold-out (top-4)\n"
        "percentual por linha (classe verdadeira)",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cm_agg_all_classifiers_pct.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Individuais agregados em %
    for name, y_pred in preds.items():
        cm = confusion_matrix(
            y_te.ravel(), y_pred.ravel(), labels=[0, 1], normalize="true"
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ConfusionMatrixDisplay(
            cm * 100.0, display_labels=["0 (fora)", "1 (máscara)"]
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f")
        for txt in ax.texts:
            txt.set_text(f"{txt.get_text()}%")
        ax.set_title(f"{name} — confusão agregada (%)")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        fig.tight_layout()
        fig.savefig(out_dir / f"cm_agg_{name}_pct.png", dpi=150)
        plt.close(fig)

    print("DONE", out_dir)
    print("percentuais:", out_dir / "cm_agg_all_classifiers_pct.png")


if __name__ == "__main__":
    main()
