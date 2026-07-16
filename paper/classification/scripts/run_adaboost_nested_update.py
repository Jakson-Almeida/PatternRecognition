"""Atualiza apenas AdaBoost no Passo 4 (nested CV) e faz merge nos CSVs."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_utils import load_cv_splits
from src.data_utils import RANDOM_STATE, RESULTS_DIR, load_prepared_dataset
from src.tuning import build_search_spaces, nested_tune_fold


def main() -> None:
    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()
    a_train, a_test = splits["a_train"], splits["a_test"]
    spec = build_search_spaces(random_state=RANDOM_STATE)["AdaBoost"]

    rows: list[dict] = []
    params_rows: list[dict] = []

    for fold in range(len(a_train)):
        tr = np.asarray(a_train[fold], dtype=int).ravel()
        te = np.asarray(a_test[fold], dtype=int).ravel()
        t0 = time.perf_counter()
        result = nested_tune_fold(
            "AdaBoost",
            copy.deepcopy(spec["estimator"]),
            spec["param_grid"],
            x[tr],
            y[tr],
            y[tr],
            x[te],
            y[te],
        )
        dt = time.perf_counter() - t0
        m = result["metrics"]
        print(
            f"fold={fold} acc={m['accuracy']:.4f} "
            f"best={result['best_params']} ({dt:.1f}s)"
        )
        row = {
            "fold": fold,
            "classifier": "AdaBoost",
            "best_inner_accuracy": result["best_inner_accuracy"],
            "n_candidates": result["n_candidates"],
            "seconds": dt,
        }
        row.update({k: v for k, v in m.items() if k != "classifier"})
        rows.append(row)
        params_rows.append(
            {
                "fold": fold,
                "classifier": "AdaBoost",
                "best_params": json.dumps(result["best_params"]),
                "best_inner_accuracy": result["best_inner_accuracy"],
            }
        )

    fold_df = pd.DataFrame(rows)

    old_folds = pd.read_csv(RESULTS_DIR / "passo4_fold_metrics.csv")
    old_folds = old_folds[old_folds["classifier"] != "AdaBoost"]
    # Align columns: keep union
    merged_folds = pd.concat([old_folds, fold_df], ignore_index=True)
    merged_folds.to_csv(RESULTS_DIR / "passo4_fold_metrics.csv", index=False)

    old_params = pd.read_csv(RESULTS_DIR / "passo4_best_params.csv")
    old_params = old_params[old_params["classifier"] != "AdaBoost"]
    merged_params = pd.concat([old_params, pd.DataFrame(params_rows)], ignore_index=True)
    merged_params.to_csv(RESULTS_DIR / "passo4_best_params.csv", index=False)

    sum_rows = []
    for name, g in merged_folds.groupby("classifier"):
        row: dict = {"classifier": name, "n_folds": len(g)}
        for c in g.columns:
            if c in ("fold", "classifier"):
                continue
            if pd.api.types.is_numeric_dtype(g[c]):
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std())
        sum_rows.append(row)
    summary = pd.DataFrame(sum_rows).sort_values("accuracy_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "passo4_summary.csv", index=False)

    ab = summary[summary["classifier"] == "AdaBoost"].iloc[0]
    print(
        f"AdaBoost nested: acc={ab['accuracy_mean']:.4f}±{ab['accuracy_std']:.4f} "
        f"f1_w={ab['f1_weighted_mean']:.4f} f1_m={ab['f1_macro_mean']:.4f}"
    )
    print("DONE")


if __name__ == "__main__":
    main()
