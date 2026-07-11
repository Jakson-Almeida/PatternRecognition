"""Avaliação única no hold-out (após tuning no conjunto de desenvolvimento)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import predict_topk_mask
from src.cv_utils import load_cv_splits
from src.data_utils import K_DEFAULT, RESULTS_DIR, load_prepared_dataset
from src.metrics_utils import evaluate_multilabel
from src.tuned_models import build_tuned_classifiers


def main() -> None:
    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_mask"], dtype=int)
    splits = load_cv_splits()
    dev = np.asarray(splits["dev_idx"], dtype=int).ravel()
    ho = np.asarray(splits["holdout_idx"], dtype=int).ravel()
    assert len(np.intersect1d(dev, ho)) == 0

    rows = []
    for name, est in build_tuned_classifiers().items():
        est.fit(x[dev], y[dev])
        pred = predict_topk_mask(est, x[ho], k=K_DEFAULT)
        m = evaluate_multilabel(y[ho], pred)
        m["classifier"] = name
        rows.append(m)
        print(
            f"{name:12s} jaccard={m['jaccard_samples']:.4f} "
            f"exact={m['exact_match']:.4f} f1={m['f1_micro']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("jaccard_samples", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "passo7_holdout_metrics.csv", index=False)
    meta = {
        "n_dev": int(len(dev)),
        "n_holdout": int(len(ho)),
        "k": int(K_DEFAULT),
        "protocol": "fit on full development set; single evaluation on hold-out",
        "models": "Passo 4 consensus hyperparameters",
        "features": "normalized optical powers only (13D)",
    }
    with open(RESULTS_DIR / "passo7_holdout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(df[["classifier", "jaccard_samples", "exact_match", "f1_micro", "hamming_loss"]].to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
