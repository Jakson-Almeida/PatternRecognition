"""Executa Passo 2: hold-out + estratégias A/B de CV (dados reais)."""

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
from src.data_utils import FIGURES_DIR, RESULTS_DIR, load_prepared_dataset


def _bin_hist_compare(target, bins, dev_idx, holdout_idx, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(target[dev_idx], bins=30, color="#4c72b0", edgecolor="white", alpha=0.9, label="dev")
    axes[0].hist(
        target[holdout_idx], bins=30, color="#c44e52", edgecolor="white", alpha=0.7, label="hold-out"
    )
    axes[0].set_xlabel(r"$\lambda_{res}$ (nm)")
    axes[0].set_ylabel("Contagem")
    axes[0].set_title("Dev vs hold-out")
    axes[0].legend(fontsize=8)

    # Fração por bin de estratificação
    all_b = np.unique(bins)
    dev_frac = np.array([(bins[dev_idx] == b).mean() for b in all_b])
    ho_frac = np.array([(bins[holdout_idx] == b).mean() for b in all_b])
    x = np.arange(len(all_b))
    w = 0.4
    axes[1].bar(x - w / 2, dev_frac, width=w, label="dev", color="#4c72b0")
    axes[1].bar(x + w / 2, ho_frac, width=w, label="hold-out", color="#c44e52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(b)) for b in all_b])
    axes[1].set_xlabel("Bin de estratificação (quantil)")
    axes[1].set_ylabel("Fração")
    axes[1].set_title("Proporção por bin")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _strategy_pos_heatmap(summary: pd.DataFrame, title: str, out_path: Path) -> None:
    cols = [c for c in summary.columns if c.startswith("test_pos_fbg")]
    mat = summary[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.28 * len(summary) + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(13))
    ax.set_xticklabels([str(i) for i in range(13)])
    ax.set_xlabel("FBG")
    if "repeat" in summary.columns and summary["repeat"].nunique() > 1:
        ylabels = [f"r{int(r)}-f{int(f)}" for r, f in zip(summary["repeat"], summary["fold"])]
    else:
        ylabels = [f"fold {int(f)}" for f in summary["fold"]]
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="n positivos no teste")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = data["X"]
    y_mask = data["y_mask"]
    target = data["target"]
    print("=== Dataset preparado ===")
    print(f"X={x.shape}, y_mask={y_mask.shape}, lambda=[{target.min():.3f}, {target.max():.3f}]")

    config = SplitConfig()  # defaults documentados
    print("=== Config ===")
    print(config)

    out = build_and_save_splits(target, y_mask, config=config)
    meta = out["meta"]

    print("=== Hold-out ===")
    print(f"n_total={meta['n_total']}, n_dev={meta['n_dev']}, n_holdout={meta['n_holdout']}")
    print(f"bins efetivos={meta['n_lambda_bins_effective']}, máscaras únicas={meta['n_unique_masks']}")
    print("dev_bin_counts:", meta["dev_bin_counts"])
    print("holdout_bin_counts:", meta["holdout_bin_counts"])

    # Checagens extras
    dev_idx, holdout_idx = out["dev_idx"], out["holdout_idx"]
    assert len(np.intersect1d(dev_idx, holdout_idx)) == 0
    assert len(dev_idx) + len(holdout_idx) == len(target)

    # Fração positiva global vs hold-out / dev
    global_frac = y_mask.mean(axis=0)
    dev_frac = y_mask[dev_idx].mean(axis=0)
    ho_frac = y_mask[holdout_idx].mean(axis=0)
    print("=== Fração positiva por FBG (global / dev / hold-out) ===")
    bal = pd.DataFrame(
        {
            "fbg": np.arange(13),
            "global": np.round(global_frac, 4),
            "dev": np.round(dev_frac, 4),
            "holdout": np.round(ho_frac, 4),
            "abs_diff_ho_global": np.round(np.abs(ho_frac - global_frac), 4),
        }
    )
    print(bal.to_string(index=False))
    bal.to_csv(RESULTS_DIR / "passo2_holdout_mask_balance.csv", index=False)

    print("=== Estratégia A (StratifiedKFold) ===")
    print(out["summary_a"][["fold", "n_train", "n_test", "lambda_mean_train", "lambda_mean_test"]].to_string(index=False))
    print("min positivos no teste (qualquer FBG):", meta["strategy_a"]["min_test_pos_any_fbg"])

    print("=== Estratégia B (RepeatedStratifiedKFold) — resumo ===")
    sb = out["summary_b"]
    print(f"n_evaluations={len(sb)}")
    print(
        "n_test: min/mean/max = "
        f"{sb['n_test'].min()} / {sb['n_test'].mean():.1f} / {sb['n_test'].max()}"
    )
    print(
        "lambda_mean_test: min/mean/max = "
        f"{sb['lambda_mean_test'].min():.3f} / {sb['lambda_mean_test'].mean():.3f} / "
        f"{sb['lambda_mean_test'].max():.3f}"
    )
    print("min positivos no teste (qualquer FBG):", meta["strategy_b"]["min_test_pos_any_fbg"])

    # Variabilidade entre folds: desvio da média de lambda_res no teste
    std_a = out["summary_a"]["lambda_mean_test"].std(ddof=0)
    std_b = sb["lambda_mean_test"].std(ddof=0)
    print(f"std(lambda_mean_test) A={std_a:.4f}  B={std_b:.4f}")

    # FBG 12 (mais raro) — cobertura no teste
    print("FBG12 test_pos — A:", out["summary_a"]["test_pos_fbg12"].tolist())
    print(
        "FBG12 test_pos — B: min/median/max = "
        f"{sb['test_pos_fbg12'].min()} / {sb['test_pos_fbg12'].median()} / {sb['test_pos_fbg12'].max()}"
    )

    _bin_hist_compare(
        target,
        out["lambda_bins"],
        dev_idx,
        holdout_idx,
        FIGURES_DIR / "passo2_holdout_vs_dev.png",
    )
    _strategy_pos_heatmap(
        out["summary_a"],
        "Estratégia A — positivos no teste por FBG",
        FIGURES_DIR / "passo2_strategy_a_pos_heatmap.png",
    )
    # Para B, heatmap completo fica alto (25 linhas) — ok
    _strategy_pos_heatmap(
        sb,
        "Estratégia B — positivos no teste por FBG (todas as repetições)",
        FIGURES_DIR / "passo2_strategy_b_pos_heatmap.png",
    )

    print("Salvo:", out["splits_path"])
    print("Salvo:", out["meta_path"])
    print("DONE — coherence_ok =", meta["coherence_ok"])


if __name__ == "__main__":
    main()
