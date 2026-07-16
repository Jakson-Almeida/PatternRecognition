"""Passo 4: nested CV multiclasse — GridSearch no treino; hold-out intacto."""

from __future__ import annotations

import copy
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

from src.cv_utils import load_cv_splits
from src.data_utils import FIGURES_DIR, N_CLASSES, RANDOM_STATE, RESULTS_DIR, load_prepared_dataset
from src.tuning import build_search_spaces, nested_tune_fold


def _as_index_array(obj) -> np.ndarray:
    return np.asarray(obj, dtype=int).ravel()


def _grid_size(param_grid: dict) -> list:
    from itertools import product

    keys = list(param_grid)
    vals = [param_grid[k] for k in keys]
    return list(product(*vals))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()

    holdout = set(_as_index_array(splits["holdout_idx"]).tolist())
    a_train = splits["a_train"]
    a_test = splits["a_test"]
    n_folds = len(a_train)
    spaces = build_search_spaces(random_state=RANDOM_STATE)

    print(f"Nested CV: outer={n_folds}, inner=3, n_classes={N_CLASSES}")
    print(f"Hold-out={len(holdout)} (não usado)")
    print("Grades:", {n: len(_grid_size(s["param_grid"])) for n, s in spaces.items()})

    rows: list[dict] = []
    params_rows: list[dict] = []

    for fold in range(n_folds):
        tr = _as_index_array(a_train[fold])
        te = _as_index_array(a_test[fold])
        if set(tr) & holdout or set(te) & holdout:
            raise AssertionError(f"Fold {fold} intersecta hold-out")
        if set(tr) & set(te):
            raise AssertionError(f"Fold {fold} vazamento treino/teste")

        x_tr, y_tr = x[tr], y[tr]
        x_te, y_te = x[te], y[te]
        strat_tr = y_tr
        print(f"\n=== Outer fold {fold}: train={len(tr)}, test={len(te)} ===")

        for name, spec in spaces.items():
            t0 = time.perf_counter()
            est = copy.deepcopy(spec["estimator"])
            result = nested_tune_fold(
                name,
                est,
                spec["param_grid"],
                x_tr,
                y_tr,
                strat_tr,
                x_te,
                y_te,
                inner_splits=3,
                random_state=RANDOM_STATE,
            )
            dt = time.perf_counter() - t0
            m = result["metrics"]
            row = {
                "fold": fold,
                "classifier": name,
                "best_inner_accuracy": result["best_inner_accuracy"],
                "n_candidates": result["n_candidates"],
                "seconds": dt,
                **{k: m[k] for k in m if k != "classifier"},
            }
            rows.append(row)
            params_rows.append(
                {
                    "fold": fold,
                    "classifier": name,
                    "best_params": json.dumps(result["best_params"], sort_keys=True),
                    "best_inner_accuracy": result["best_inner_accuracy"],
                }
            )
            print(
                f"  {name:12s}  acc={m['accuracy']:.4f}  "
                f"f1_w={m['f1_weighted']:.4f}  "
                f"inner={result['best_inner_accuracy']:.4f}  "
                f"params={result['best_params']}  ({dt:.1f}s)"
            )

    df = pd.DataFrame(rows)
    params_df = pd.DataFrame(params_rows)

    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "best_inner_accuracy",
        "seconds",
    ]
    summary_rows = []
    for name, g in df.groupby("classifier", sort=False):
        row = {"classifier": name, "n_folds": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "accuracy_mean", ascending=False
    ).reset_index(drop=True)

    passo3_path = RESULTS_DIR / "passo3_summary.csv"
    if passo3_path.exists():
        p3 = pd.read_csv(passo3_path)[["classifier", "accuracy_mean"]].rename(
            columns={"accuracy_mean": "accuracy_passo3"}
        )
        summary = summary.merge(p3, on="classifier", how="left")
        summary["accuracy_delta"] = summary["accuracy_mean"] - summary["accuracy_passo3"]

    fold_path = RESULTS_DIR / "passo4_fold_metrics.csv"
    summary_path = RESULTS_DIR / "passo4_summary.csv"
    params_path = RESULTS_DIR / "passo4_best_params.csv"
    df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    params_df.to_csv(params_path, index=False)

    meta = {
        "protocol": "nested CV",
        "outer": "strategy A StratifiedKFold by y_class (5)",
        "inner": "StratifiedKFold (3) on train fold, stratify=y_class",
        "scoring": "accuracy",
        "holdout_used": False,
        "n_classes": int(N_CLASSES),
        "random_state": int(RANDOM_STATE),
        "mq_note": "RidgeClassifier (linear multiclass baseline)",
        "ranking_by_accuracy_mean": summary["classifier"].tolist(),
    }
    meta_path = RESULTS_DIR / "passo4_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Resumo nested (média ± std no teste externo) ===")
    cols = [
        "classifier",
        "accuracy_mean",
        "accuracy_std",
        "f1_weighted_mean",
        "f1_macro_mean",
        "seconds_mean",
    ]
    if "accuracy_delta" in summary.columns:
        cols += ["accuracy_passo3", "accuracy_delta"]
    show = summary[cols].copy()
    for c in show.columns:
        if c != "classifier":
            show[c] = show[c].map(lambda v: round(float(v), 4))
    print(show.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = summary["classifier"].tolist()
    x_pos = np.arange(len(names))
    if "accuracy_passo3" in summary.columns:
        w = 0.35
        ax.bar(x_pos - w / 2, summary["accuracy_passo3"], w, label="Passo 3 (default)", color="#a0a0a0")
        ax.bar(
            x_pos + w / 2,
            summary["accuracy_mean"],
            w,
            yerr=summary["accuracy_std"],
            label="Passo 4 (tuned)",
            color="#4c72b0",
            capsize=3,
        )
    else:
        ax.bar(x_pos, summary["accuracy_mean"], yerr=summary["accuracy_std"], color="#4c72b0", capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acurácia")
    ax.set_title("Passo 4 vs Passo 3 — acurácia (CV)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "passo4_vs_passo3_jaccard.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"Salvo: {fold_path}")
    print(f"Salvo: {summary_path}")
    print(f"Salvo: {params_path}")
    print(f"Salvo: {meta_path}")
    print(f"Salvo: {fig_path}")
    print("DONE")


if __name__ == "__main__":
    main()
