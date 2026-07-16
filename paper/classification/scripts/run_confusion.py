"""Gera matrizes de confusão 10x10 no hold-out (multiclasse)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, N_CLASSES, RESULTS_DIR, load_prepared_dataset
from src.metrics_utils import evaluate_multiclass, multiclass_confusion
from src.tuned_models import build_tuned_classifiers


def plot_cm(cm, title, out_path, values_format="d"):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(N_CLASSES)]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=values_format
    )
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = FIGURES_DIR / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()
    X_tr, y_tr = X[dev], y[dev]
    X_te, y_te = X[ho], y[ho]
    print(f"dev={len(dev)} holdout={len(ho)} n_classes={N_CLASSES}")

    preds = {}
    rows = []
    for name, est in build_tuned_classifiers().items():
        print(f"fit {name}...")
        est.fit(X_tr, y_tr)
        pred = np.asarray(est.predict(X_te), dtype=int).ravel()
        preds[name] = pred
        m = evaluate_multiclass(y_te, pred)
        m["classifier"] = name
        rows.append(m)
        cm = multiclass_confusion(y_te, pred)
        plot_cm(cm, f"{name} — hold-out", out_dir / f"cm_agg_{name}.png")
        cm_pct = multiclass_confusion(y_te, pred, normalize="true") * 100.0
        plot_cm(cm_pct, f"{name} — hold-out (%)", out_dir / f"cm_agg_{name}_pct.png", values_format=".1f")
        print(name, f"acc={m['accuracy']:.4f}")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "confusion_agg_holdout.csv", index=False)

    names = list(preds.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        cm = multiclass_confusion(y_te, preds[name])
        ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(N_CLASSES)]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
    fig.suptitle("Matrizes 10×10 — hold-out (multiclasse)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cm_agg_all_classifiers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        cm = multiclass_confusion(y_te, preds[name], normalize="true") * 100.0
        ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(N_CLASSES)]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".0f"
        )
        ax.set_title(name)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
    fig.suptitle("Matrizes 10×10 (%) — hold-out\npercentual por linha", y=1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "cm_agg_all_classifiers_pct.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("DONE", out_dir)


if __name__ == "__main__":
    main()
