"""Executa Passo 2: hold-out + CV estratificados por y_class."""

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

from src.cv_utils import SplitConfig, build_and_save_splits
from src.data_utils import FIGURES_DIR, N_CLASSES, RESULTS_DIR, load_prepared_dataset


def _bin_hist_compare(target, y_class, dev_idx, holdout_idx, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(target[dev_idx], bins=30, color="#4c72b0", edgecolor="white", alpha=0.9, label="dev")
    axes[0].hist(
        target[holdout_idx], bins=30, color="#c44e52", edgecolor="white", alpha=0.7, label="hold-out"
    )
    axes[0].set_xlabel(r"$\lambda_{res}$ (nm)")
    axes[0].set_ylabel("Contagem")
    axes[0].set_title("Dev vs hold-out (cobertura espectral)")
    axes[0].legend(fontsize=8)

    classes = np.arange(N_CLASSES)
    dev_frac = np.array([(y_class[dev_idx] == c).mean() for c in classes])
    ho_frac = np.array([(y_class[holdout_idx] == c).mean() for c in classes])
    x = np.arange(N_CLASSES)
    w = 0.4
    axes[1].bar(x - w / 2, dev_frac, width=w, label="dev", color="#4c72b0")
    axes[1].bar(x + w / 2, ho_frac, width=w, label="hold-out", color="#c44e52")
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Classe")
    axes[1].set_ylabel("Fração")
    axes[1].set_title("Proporção por classe (estratificação)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _strategy_class_heatmap(summary: pd.DataFrame, title: str, out_path: Path) -> None:
    cols = [c for c in summary.columns if c.startswith("test_class_")]
    mat = summary[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.28 * len(summary) + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("test_class_", "C") for c in cols])
    ax.set_xlabel("Classe")
    if "repeat" in summary.columns and summary["repeat"].nunique() > 1:
        ylabels = [f"r{int(r)}-f{int(f)}" for r, f in zip(summary["repeat"], summary["fold"])]
    else:
        ylabels = [f"fold {int(f)}" for f in summary["fold"]]
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="n no teste")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = data["X"]
    y_class = np.asarray(data["y_class"], dtype=int).ravel()
    target = data["target"]
    print("=== Dataset preparado ===")
    print(f"X={x.shape}, y_class={y_class.shape}, n_classes={N_CLASSES}")
    print(f"lambda=[{target.min():.3f}, {target.max():.3f}]")

    config = SplitConfig()
    print("=== Config ===")
    print(config)

    out = build_and_save_splits(target, y_class, config=config)
    meta = out["meta"]

    print("=== Hold-out ===")
    print(f"n_total={meta['n_total']}, n_dev={meta['n_dev']}, n_holdout={meta['n_holdout']}")
    print("dev_class_counts:", meta["dev_class_counts"])
    print("holdout_class_counts:", meta["holdout_class_counts"])
    print("min_test_class A/B:", meta["strategy_a"]["min_test_count_any_class"], meta["strategy_b"]["min_test_count_any_class"])

    dev_idx, holdout_idx = out["dev_idx"], out["holdout_idx"]
    assert len(np.intersect1d(dev_idx, holdout_idx)) == 0
    assert len(dev_idx) + len(holdout_idx) == len(target)

    global_frac = np.bincount(y_class, minlength=N_CLASSES) / len(y_class)
    dev_frac = np.bincount(y_class[dev_idx], minlength=N_CLASSES) / len(dev_idx)
    ho_frac = np.bincount(y_class[holdout_idx], minlength=N_CLASSES) / len(holdout_idx)
    bal = pd.DataFrame(
        {
            "class": np.arange(N_CLASSES),
            "global": np.round(global_frac, 4),
            "dev": np.round(dev_frac, 4),
            "holdout": np.round(ho_frac, 4),
            "abs_diff_ho_global": np.round(np.abs(ho_frac - global_frac), 4),
        }
    )
    print(bal.to_string(index=False))
    bal.to_csv(RESULTS_DIR / "passo2_holdout_mask_balance.csv", index=False)

    print("=== Estratégia A ===")
    print(out["summary_a"][["fold", "n_train", "n_test", "lambda_mean_train", "lambda_mean_test"]].to_string(index=False))
    assert meta["strategy_a"]["min_test_count_any_class"] >= 1

    _bin_hist_compare(
        target, y_class, dev_idx, holdout_idx, FIGURES_DIR / "passo2_holdout_vs_dev.png"
    )
    _strategy_class_heatmap(
        out["summary_a"],
        "Estratégia A — contagem por classe no teste",
        FIGURES_DIR / "passo2_strategy_a_pos_heatmap.png",
    )
    _strategy_class_heatmap(
        out["summary_b"],
        "Estratégia B — contagem por classe no teste",
        FIGURES_DIR / "passo2_strategy_b_pos_heatmap.png",
    )
    print("DONE", out["splits_path"])


if __name__ == "__main__":
    main()
