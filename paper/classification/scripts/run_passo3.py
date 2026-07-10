"""Passo 3: avalia os 6 classificadores na CV estratégia A (hold-out intacto)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import build_classifiers, fit_predict_fold
from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, K_DEFAULT, RESULTS_DIR, load_prepared_dataset


def _as_index_array(obj) -> np.ndarray:
    return np.asarray(obj, dtype=int).ravel()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_mask"], dtype=int)
    splits = load_cv_splits()

    holdout = set(_as_index_array(splits["holdout_idx"]).tolist())
    a_train = splits["a_train"]
    a_test = splits["a_test"]
    n_folds = len(a_train)
    print(f"Dataset X={x.shape}, folds A={n_folds}, k={K_DEFAULT}")
    print(f"Hold-out size={len(holdout)} (não usado neste passo)")

    rows: list[dict] = []
    fold_times: list[dict] = []

    for fold in range(n_folds):
        tr = _as_index_array(a_train[fold])
        te = _as_index_array(a_test[fold])
        if set(tr).intersection(holdout) or set(te).intersection(holdout):
            raise AssertionError(f"Fold {fold} intersecta hold-out")
        if set(tr).intersection(te):
            raise AssertionError(f"Fold {fold} com vazamento treino/teste")

        x_tr, y_tr = x[tr], y[tr]
        x_te, y_te = x[te], y[te]
        print(f"\n=== Fold {fold}: train={len(tr)}, test={len(te)} ===")

        # Novas instâncias por fold (sem estado entre folds)
        models = build_classifiers()
        for name, est in models.items():
            t0 = time.perf_counter()
            metrics = fit_predict_fold(name, est, x_tr, y_tr, x_te, y_te, k=K_DEFAULT)
            dt = time.perf_counter() - t0
            metrics["fold"] = fold
            metrics["seconds"] = dt
            rows.append(metrics)
            fold_times.append({"fold": fold, "classifier": name, "seconds": dt})
            print(
                f"  {name:12s}  jaccard={metrics['jaccard_samples']:.4f}  "
                f"exact={metrics['exact_match']:.4f}  "
                f"f1_micro={metrics['f1_micro']:.4f}  "
                f"set_recall={metrics['set_recall']:.4f}  ({dt:.1f}s)"
            )
            # Predição top-k deve ter exatamente k uns
            if abs(metrics["mean_pred_positives"] - K_DEFAULT) > 1e-9:
                raise AssertionError(
                    f"{name} fold {fold}: mean_pred_positives="
                    f"{metrics['mean_pred_positives']} != {K_DEFAULT}"
                )

    df = pd.DataFrame(rows)
    metric_cols = [
        "hamming_loss",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "exact_match",
        "jaccard_samples",
        "set_recall",
        "seconds",
    ]

    summary_rows = []
    for name, g in df.groupby("classifier", sort=False):
        row = {"classifier": name, "n_folds": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Ordenar por jaccard médio (métrica alinhada ao conjunto top-k)
    summary = summary.sort_values("jaccard_samples_mean", ascending=False).reset_index(drop=True)

    fold_path = RESULTS_DIR / "passo3_fold_metrics.csv"
    summary_path = RESULTS_DIR / "passo3_summary.csv"
    df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "cv_strategy": "A_StratifiedKFold",
        "n_folds": n_folds,
        "k": int(K_DEFAULT),
        "holdout_used": False,
        "prediction": "top-k from scores (exactly k positives per sample)",
        "classifiers": list(build_classifiers().keys()),
        "defaults": {
            "kNN": "n_neighbors=5, StandardScaler",
            "SVM": "OvR RBF C=1.0, StandardScaler",
            "MLP": "hidden=(64,32), early_stopping, StandardScaler",
            "RandomForest": "n_estimators=100, multilabel nativo",
            "AdaBoost": "OvR n_estimators=50",
            "MQ": "MultiOutput LinearRegression + StandardScaler; top-k nos valores previstos",
        },
        "ranking_by_jaccard_samples_mean": summary["classifier"].tolist(),
        "smote": False,
    }
    meta_path = RESULTS_DIR / "passo3_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Resumo (média ± std entre folds) ===")
    show = summary[
        [
            "classifier",
            "jaccard_samples_mean",
            "jaccard_samples_std",
            "exact_match_mean",
            "f1_micro_mean",
            "f1_macro_mean",
            "hamming_loss_mean",
            "set_recall_mean",
            "seconds_mean",
        ]
    ].copy()
    for c in show.columns:
        if c != "classifier":
            show[c] = show[c].map(lambda v: round(float(v), 4))
    print(show.to_string(index=False))

    # Figura: barras das métricas principais
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = summary["classifier"].tolist()
    x_pos = np.arange(len(names))
    w = 0.25
    ax.bar(
        x_pos - w,
        summary["jaccard_samples_mean"],
        w,
        yerr=summary["jaccard_samples_std"],
        label="Jaccard (samples)",
        color="#4c72b0",
        capsize=3,
    )
    ax.bar(
        x_pos,
        summary["f1_micro_mean"],
        w,
        yerr=summary["f1_micro_std"],
        label="F1 micro",
        color="#55a868",
        capsize=3,
    )
    ax.bar(
        x_pos + w,
        summary["exact_match_mean"],
        w,
        yerr=summary["exact_match_std"],
        label="Exact match",
        color="#c44e52",
        capsize=3,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Passo 3 — CV estratégia A (média ± std)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "passo3_metrics_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\nSalvo: {fold_path}")
    print(f"Salvo: {summary_path}")
    print(f"Salvo: {meta_path}")
    print(f"Salvo: {fig_path}")
    print("DONE")


if __name__ == "__main__":
    main()
