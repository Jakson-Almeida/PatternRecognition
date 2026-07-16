"""Passo 5: análise multiclasse — erros vs lambda_res e +wl_bragg."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, N_CLASSES, RESULTS_DIR, load_prepared_dataset
from src.metrics_utils import evaluate_multiclass
from src.tuned_models import build_tuned_classifiers


def _as_index(a):
    return np.asarray(a, dtype=int).ravel()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    target = np.asarray(data["target"], dtype=float).ravel()
    wl = np.asarray(data["wl_bragg"], dtype=float)
    splits = load_cv_splits()
    a_train, a_test = splits["a_train"], splits["a_test"]

    # CV stacked predictions for best model (SVM) vs lambda
    y_true_all, y_pred_all, lam_all = [], [], []
    for fold in range(len(a_train)):
        tr, te = _as_index(a_train[fold]), _as_index(a_test[fold])
        clf = build_tuned_classifiers()["SVM"]
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(pred)
        lam_all.append(target[te])
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    lam_all = np.concatenate(lam_all)

    bins = pd.qcut(lam_all, q=8, duplicates="drop")
    rows = []
    for b in bins.categories:
        mask = bins == b
        if mask.sum() == 0:
            continue
        m = evaluate_multiclass(y_true_all[mask], y_pred_all[mask])
        rows.append({"bin": str(b), "n": int(mask.sum()), "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"]})
    err_df = pd.DataFrame(rows)
    err_df.to_csv(RESULTS_DIR / "passo5_errors_vs_lambda.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(err_df)), err_df["accuracy"], color="#4c72b0", edgecolor="white")
    ax.set_xticks(range(len(err_df)))
    ax.set_xticklabels(err_df["bin"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Acurácia (SVM)")
    ax.set_title(r"Acurácia vs bins de $\lambda_{res}$ (CV empilhada)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_jaccard_vs_lambda.png", dpi=150)
    fig.savefig(FIGURES_DIR / "fig_jaccard_vs_lambda.png", dpi=150)
    plt.close(fig)

    # Feature ablation: powers vs powers+wl
    deltas = []
    for name, factory in build_tuned_classifiers().items():
        acc_p, acc_w = [], []
        for fold in range(len(a_train)):
            tr, te = _as_index(a_train[fold]), _as_index(a_test[fold])
            # powers only
            est = build_tuned_classifiers()[name]
            est.fit(X[tr], y[tr])
            acc_p.append(evaluate_multiclass(y[te], est.predict(X[te]))["accuracy"])
            # + wl (z-score wl on train)
            wl_tr = wl[tr]
            mu, sd = wl_tr.mean(axis=0), wl_tr.std(axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            Xw_tr = np.hstack([X[tr], (wl[tr] - mu) / sd])
            Xw_te = np.hstack([X[te], (wl[te] - mu) / sd])
            est2 = build_tuned_classifiers()[name]
            est2.fit(Xw_tr, y[tr])
            acc_w.append(evaluate_multiclass(y[te], est2.predict(Xw_te))["accuracy"])
        deltas.append(
            {
                "classifier": name,
                "powers_only": float(np.mean(acc_p)),
                "powers_plus_wl": float(np.mean(acc_w)),
                "delta_wl": float(np.mean(acc_w) - np.mean(acc_p)),
            }
        )
        print(name, deltas[-1])
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(RESULTS_DIR / "passo5_feature_delta.csv", index=False)
    delta_df.to_csv(RESULTS_DIR / "passo5_compare_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    xpos = np.arange(len(delta_df))
    w = 0.35
    ax.bar(xpos - w / 2, delta_df["powers_only"], w, label="só potências", color="#4c72b0")
    ax.bar(xpos + w / 2, delta_df["powers_plus_wl"], w, label="+ wl_bragg", color="#55a868")
    ax.set_xticks(xpos)
    ax.set_xticklabels(delta_df["classifier"], rotation=15)
    ax.set_ylabel("Acurácia CV")
    ax.set_title("Efeito de incluir wl_bragg")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_compare_features.png", dpi=150)
    plt.close(fig)

    # Tuned metrics bar (from passo4)
    s4 = pd.read_csv(RESULTS_DIR / "passo4_summary.csv").sort_values("accuracy_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(s4["classifier"], s4["accuracy_mean"], yerr=s4["accuracy_std"], color="#4c72b0", capsize=3)
    ax.set_ylabel("Acurácia")
    ax.set_title("Passo 5 — modelos afinados (CV)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_metrics_tuned.png", dpi=150)
    fig.savefig(FIGURES_DIR / "fig_metrics_classifiers.png", dpi=150)
    plt.close(fig)

    # Save k4 summary alias
    s4.to_csv(RESULTS_DIR / "passo5_summary_k4.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
