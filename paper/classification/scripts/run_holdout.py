"""Avaliação final no hold-out com modelos afinados (multiclasse)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_utils import load_cv_splits
from src.data_utils import RESULTS_DIR, load_prepared_dataset
from src.metrics_utils import evaluate_multiclass
from src.tuned_models import build_tuned_classifiers


def main() -> None:
    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_class"], dtype=int).ravel()
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()
    assert len(np.intersect1d(dev, ho)) == 0

    rows = []
    for name, est in build_tuned_classifiers().items():
        est.fit(x[dev], y[dev])
        pred = np.asarray(est.predict(x[ho]), dtype=int).ravel()
        m = evaluate_multiclass(y[ho], pred)
        m["classifier"] = name
        rows.append(m)
        print(f"{name:12s} acc={m['accuracy']:.4f} f1_w={m['f1_weighted']:.4f} f1_m={m['f1_macro']:.4f}")

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    out = RESULTS_DIR / "passo7_holdout_metrics.csv"
    df.to_csv(out, index=False)
    print(df[["classifier", "accuracy", "f1_weighted", "f1_macro", "precision_weighted"]].to_string(index=False))
    print("Salvo", out)


if __name__ == "__main__":
    main()
