"""Passo 4: nested CV — GridSearch no treino de cada fold A; hold-out intacto."""

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
from src.data_utils import FIGURES_DIR, K_DEFAULT, RANDOM_STATE, RESULTS_DIR, load_prepared_dataset
from src.tuning import build_search_spaces, nested_tune_fold


def _as_index_array(obj) -> np.ndarray:
    return np.asarray(obj, dtype=int).ravel()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_mask"], dtype=int)
    splits = load_cv_splits()
    bins = np.asarray(splits["lambda_bins"], dtype=int).ravel()

    holdout = set(_as_index_array(splits["holdout_idx"]).tolist())
    a_train = splits["a_train"]
    a_test = splits["a_test"]
    n_folds = len(a_train)
    spaces = build_search_spaces(random_state=RANDOM_STATE)

    print(f"Nested CV: outer={n_folds} (estratégia A), inner=3, k={K_DEFAULT}")
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
        strat_tr = bins[tr]
        print(f"\n=== Outer fold {fold}: train={len(tr)}, test={len(te)} ===")

        for name, spec in spaces.items():
            t0 = time.perf_counter()
            # clone fresco por fold
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
                k=K_DEFAULT,
            )
            dt = time.perf_counter() - t0
            m = result["metrics"]
            row = {
                "fold": fold,
                "classifier": name,
                "best_inner_jaccard": result["best_inner_jaccard"],
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
                    "best_inner_jaccard": result["best_inner_jaccard"],
                }
            )
            print(
                f"  {name:12s}  jaccard={m['jaccard_samples']:.4f}  "
                f"exact={m['exact_match']:.4f}  "
                f"inner={result['best_inner_jaccard']:.4f}  "
                f"params={result['best_params']}  ({dt:.1f}s)"
            )

    df = pd.DataFrame(rows)
    params_df = pd.DataFrame(params_rows)

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
        "best_inner_jaccard",
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
        "jaccard_samples_mean", ascending=False
    ).reset_index(drop=True)

    # Comparar com Passo 3 se existir
    passo3_path = RESULTS_DIR / "passo3_summary.csv"
    if passo3_path.exists():
        p3 = pd.read_csv(passo3_path)[["classifier", "jaccard_samples_mean"]].rename(
            columns={"jaccard_samples_mean": "jaccard_passo3"}
        )
        summary = summary.merge(p3, on="classifier", how="left")
        summary["jaccard_delta"] = summary["jaccard_samples_mean"] - summary["jaccard_passo3"]

    fold_path = RESULTS_DIR / "passo4_fold_metrics.csv"
    summary_path = RESULTS_DIR / "passo4_summary.csv"
    params_path = RESULTS_DIR / "passo4_best_params.csv"
    df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    params_df.to_csv(params_path, index=False)

    meta = {
        "protocol": "nested CV",
        "outer": "strategy A StratifiedKFold (5)",
        "inner": "StratifiedKFold (3) on train fold, stratify=lambda_bins",
        "scoring": "topk Jaccard samples",
        "holdout_used": False,
        "k": int(K_DEFAULT),
        "random_state": int(RANDOM_STATE),
        "mq_note": "Passo 4 busca alpha de Ridge; Passo 3 usava LinearRegression (OLS)",
        "ranking_by_jaccard_samples_mean": summary["classifier"].tolist(),
    }
    meta_path = RESULTS_DIR / "passo4_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Resumo nested (média ± std no teste externo) ===")
    cols = [
        "classifier",
        "jaccard_samples_mean",
        "jaccard_samples_std",
        "exact_match_mean",
        "f1_micro_mean",
        "seconds_mean",
    ]
    if "jaccard_delta" in summary.columns:
        cols += ["jaccard_passo3", "jaccard_delta"]
    show = summary[cols].copy()
    for c in show.columns:
        if c != "classifier":
            show[c] = show[c].map(lambda v: round(float(v), 4) if pd.notna(v) else v)
    print(show.to_string(index=False))

    # Figura
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = summary["classifier"].tolist()
    xpos = np.arange(len(names))
    w = 0.35
    ax.bar(
        xpos - w / 2,
        summary["jaccard_samples_mean"],
        w,
        yerr=summary["jaccard_samples_std"],
        label="Passo 4 (tuned)",
        color="#4c72b0",
        capsize=3,
    )
    if "jaccard_passo3" in summary.columns:
        ax.bar(
            xpos + w / 2,
            summary["jaccard_passo3"],
            w,
            label="Passo 3 (default)",
            color="#ccb974",
        )
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Jaccard (samples)")
    ax.set_title("Passo 4 nested CV vs Passo 3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "passo4_vs_passo3_jaccard.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\nSalvo: {fold_path}")
    print(f"Salvo: {summary_path}")
    print(f"Salvo: {params_path}")
    print(f"Salvo: {meta_path}")
    print(f"Salvo: {fig_path}")
    print("DONE")


def _grid_size(param_grid: dict) -> list:
    """Placeholder para print — retorna lista de combinações aproximada."""
    from itertools import product

    keys = list(param_grid)
    vals = [param_grid[k] for k in keys]
    return list(product(*vals))


if __name__ == "__main__":
    main()
