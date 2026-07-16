"""Comparação AdaBoost antigo (stump) vs AdaBoost adaptado ao multiclasse."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_utils import load_cv_splits
from src.data_utils import RANDOM_STATE, RESULTS_DIR, load_prepared_dataset
from src.metrics_utils import evaluate_multiclass


def make_old_adaboost(random_state: int = RANDOM_STATE) -> AdaBoostClassifier:
    """Configuração anterior: stump depth=1, lr=1.0, n=50."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
        n_estimators=50,
        learning_rate=1.0,
        random_state=random_state,
    )


def make_new_adaboost(
    random_state: int = RANDOM_STATE,
    *,
    max_depth: int = 3,
    n_estimators: int = 200,
    learning_rate: float = 0.5,
) -> AdaBoostClassifier:
    """AdaBoost com árvore mais profunda e classes balanceadas."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
        ),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def evaluate_on_cv(name: str, factory, X, y, a_train, a_test) -> list[dict]:
    rows = []
    for fold in range(len(a_train)):
        tr = np.asarray(a_train[fold], dtype=int).ravel()
        te = np.asarray(a_test[fold], dtype=int).ravel()
        est = factory()
        t0 = time.perf_counter()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        dt = time.perf_counter() - t0
        m = evaluate_multiclass(y[te], pred)
        m.update({"variant": name, "fold": fold, "seconds": dt})
        rows.append(m)
        print(f"  {name:12s} fold={fold} acc={m['accuracy']:.4f} f1_m={m['f1_macro']:.4f} ({dt:.1f}s)")
    return rows


def tune_new_on_dev(X_dev, y_dev) -> dict:
    """Grid compacta só no desenvolvimento (3-fold interno)."""
    base = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    grid = {
        "estimator__max_depth": [2, 3, 4],
        "n_estimators": [100, 200],
        "learning_rate": [0.3, 0.5, 1.0],
    }
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        base,
        grid,
        scoring="accuracy",
        cv=inner,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_dev, y_dev)
    return {
        "best_params": search.best_params_,
        "best_cv_accuracy": float(search.best_score_),
        "best_estimator": search.best_estimator_,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_prepared_dataset()
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()
    a_train, a_test = splits["a_train"], splits["a_test"]

    print("=== A) CV estratégia A: old vs new (defaults) ===")
    rows = []
    rows += evaluate_on_cv("old_stump", make_old_adaboost, X, y, a_train, a_test)
    rows += evaluate_on_cv("new_depth3", make_new_adaboost, X, y, a_train, a_test)

    print("\n=== B) GridSearch no desenvolvimento (só new) ===")
    t0 = time.perf_counter()
    tuned = tune_new_on_dev(X[dev], y[dev])
    print(f"best_params={tuned['best_params']} inner_acc={tuned['best_cv_accuracy']:.4f} ({time.perf_counter()-t0:.1f}s)")

    # Avaliar best params nos folds externos (refit por fold com params fixos)
    bp = tuned["best_params"]

    def make_tuned():
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=bp["estimator__max_depth"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_estimators=bp["n_estimators"],
            learning_rate=bp["learning_rate"],
            random_state=RANDOM_STATE,
        )

    rows += evaluate_on_cv("new_tuned", make_tuned, X, y, a_train, a_test)

    print("\n=== C) Hold-out ===")
    hold_rows = []
    for name, factory in [
        ("old_stump", make_old_adaboost),
        ("new_depth3", make_new_adaboost),
        ("new_tuned", make_tuned),
    ]:
        est = factory()
        est.fit(X[dev], y[dev])
        pred = est.predict(X[ho])
        m = evaluate_multiclass(y[ho], pred)
        m["variant"] = name
        hold_rows.append(m)
        print(
            f"  {name:12s} holdout acc={m['accuracy']:.4f} "
            f"f1_w={m['f1_weighted']:.4f} f1_m={m['f1_macro']:.4f}"
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("variant")[["accuracy", "f1_weighted", "f1_macro", "seconds"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    # flatten columns
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary = summary.sort_values("accuracy_mean", ascending=False)

    out_fold = RESULTS_DIR / "adaboost_multiclass_ablation_folds.csv"
    out_sum = RESULTS_DIR / "adaboost_multiclass_ablation_summary.csv"
    out_hold = RESULTS_DIR / "adaboost_multiclass_ablation_holdout.csv"
    df.to_csv(out_fold, index=False)
    summary.to_csv(out_sum, index=False)
    pd.DataFrame(hold_rows).to_csv(out_hold, index=False)

    meta = {
        "old": "DecisionTree max_depth=1, n_estimators=50, lr=1.0",
        "new_depth3": "DecisionTree max_depth=3 + class_weight=balanced, n=200, lr=0.5",
        "new_tuned_best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, float) else v) for k, v in bp.items()},
        "new_tuned_inner_accuracy": tuned["best_cv_accuracy"],
    }
    # json-serialize params cleanly
    meta["new_tuned_best_params"] = {
        k: (int(v) if hasattr(v, "item") else v) for k, v in bp.items()
    }
    with open(RESULTS_DIR / "adaboost_multiclass_ablation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Resumo CV ===")
    print(summary.to_string(index=False))
    print(f"\nSalvo: {out_sum}")
    print("DONE")


if __name__ == "__main__":
    main()
