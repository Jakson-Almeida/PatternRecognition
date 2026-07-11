"""Passo 5: métricas detalhadas, erros vs lambda_res, k=3/4/5, com/sem posições FBG."""

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
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import predict_topk_mask
from src.cv_utils import load_cv_splits
from src.data_utils import (
    FIGURES_DIR,
    K_DEFAULT,
    RESULTS_DIR,
    load_prepared_dataset,
    make_topk_mask,
)
from src.metrics_utils import evaluate_multilabel
from src.tuned_models import build_tuned_classifiers


def _as_index_array(obj) -> np.ndarray:
    return np.asarray(obj, dtype=int).ravel()


def _run_cv(
    name: str,
    estimator,
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    a_train,
    a_test,
    holdout: set[int],
    k: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """CV estratégia A; devolve métricas por fold e predições empilhadas do dev."""
    rows = []
    y_true_all = []
    y_pred_all = []
    target_all = []
    idx_all = []

    for fold in range(len(a_train)):
        tr = _as_index_array(a_train[fold])
        te = _as_index_array(a_test[fold])
        if set(tr) & holdout or set(te) & holdout:
            raise AssertionError(f"{name} fold {fold}: hold-out leakage")
        est = copy.deepcopy(estimator)
        est.fit(x[tr], y[tr].astype(int))
        pred = predict_topk_mask(est, x[te], k=k)
        m = evaluate_multilabel(y[te], pred)
        m.update({"fold": fold, "classifier": name, "k": k})
        rows.append(m)
        y_true_all.append(y[te])
        y_pred_all.append(pred)
        target_all.append(target[te])
        idx_all.append(te)

    pooled = {
        "y_true": np.vstack(y_true_all),
        "y_pred": np.vstack(y_pred_all),
        "target": np.concatenate(target_all),
        "idx": np.concatenate(idx_all),
    }
    return pd.DataFrame(rows), pooled


def _per_fbg_stats(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append(
            {
                "fbg": j,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support_pos": int(tp + fn),
                "error_rate": (fp + fn) / len(yt),
            }
        )
    return pd.DataFrame(rows)


def _errors_vs_lambda(
    y_true: np.ndarray, y_pred: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    # Jaccard por amostra
    inter = (y_true.astype(bool) & y_pred.astype(bool)).sum(axis=1)
    union = (y_true.astype(bool) | y_pred.astype(bool)).sum(axis=1).astype(float)
    union = np.where(union == 0, 1.0, union)
    jacc = inter / union
    exact = (y_true == y_pred).all(axis=1).astype(float)
    set_rec = inter / np.maximum(y_true.sum(axis=1), 1)

    bins = pd.qcut(target, q=n_bins, duplicates="drop")
    df = pd.DataFrame(
        {
            "lambda_bin": bins.astype(str),
            "lambda_mid": target,
            "jaccard": jacc,
            "exact": exact,
            "set_recall": set_rec,
        }
    )
    # midpoint aproximado por média do bin
    g = df.groupby("lambda_bin", observed=True).agg(
        n=("jaccard", "size"),
        lambda_mean=("lambda_mid", "mean"),
        jaccard_mean=("jaccard", "mean"),
        exact_mean=("exact", "mean"),
        set_recall_mean=("set_recall", "mean"),
    )
    return g.reset_index().sort_values("lambda_mean")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    wl = np.asarray(data["wl_bragg"], dtype=float)
    target = np.asarray(data["target"], dtype=float).ravel()
    splits = load_cv_splits()
    holdout = set(_as_index_array(splits["holdout_idx"]).tolist())
    a_train, a_test = splits["a_train"], splits["a_test"]

    print(f"Passo 5 | n={len(target)} | hold-out intacto ({len(holdout)})")
    models = build_tuned_classifiers()

    # ---------- A) k=4, só potências: métricas + análise ----------
    print("\n=== A) k=4, X = potências (modelos afinados) ===")
    y4 = np.asarray(data["y_mask"], dtype=int)
    fold_rows = []
    pooled_by_clf = {}
    for name, est in models.items():
        t0 = time.perf_counter()
        fold_df, pooled = _run_cv(
            name, est, x, y4, target, a_train, a_test, holdout, k=4
        )
        fold_rows.append(fold_df)
        pooled_by_clf[name] = pooled
        print(
            f"  {name:12s}  jaccard={fold_df['jaccard_samples'].mean():.4f}±"
            f"{fold_df['jaccard_samples'].std(ddof=0):.4f}  ({time.perf_counter()-t0:.1f}s)"
        )

    folds_a = pd.concat(fold_rows, ignore_index=True)
    folds_a.to_csv(RESULTS_DIR / "passo5_fold_metrics_k4.csv", index=False)

    summary_a = (
        folds_a.groupby("classifier", sort=False)
        .agg(
            jaccard_mean=("jaccard_samples", "mean"),
            jaccard_std=("jaccard_samples", lambda s: s.std(ddof=0)),
            exact_mean=("exact_match", "mean"),
            exact_std=("exact_match", lambda s: s.std(ddof=0)),
            f1_micro_mean=("f1_micro", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            hamming_mean=("hamming_loss", "mean"),
            set_recall_mean=("set_recall", "mean"),
            precision_micro_mean=("precision_micro", "mean"),
            recall_micro_mean=("recall_micro", "mean"),
            precision_macro_mean=("precision_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
        )
        .reset_index()
        .sort_values("jaccard_mean", ascending=False)
    )
    summary_a.to_csv(RESULTS_DIR / "passo5_summary_k4.csv", index=False)
    print(summary_a.to_string(index=False))

    # Melhor modelo para figuras detalhadas (maior jaccard médio)
    best_name = summary_a.iloc[0]["classifier"]
    print(f"\nModelo para análise detalhada: {best_name}")
    best_pooled = pooled_by_clf[best_name]

    # Por FBG — todos os classificadores
    fbg_tables = []
    for name, pooled in pooled_by_clf.items():
        stats = _per_fbg_stats(pooled["y_true"], pooled["y_pred"])
        stats.insert(0, "classifier", name)
        fbg_tables.append(stats)
    fbg_df = pd.concat(fbg_tables, ignore_index=True)
    fbg_df.to_csv(RESULTS_DIR / "passo5_per_fbg_metrics.csv", index=False)

    # Erros vs lambda — todos
    lam_tables = []
    for name, pooled in pooled_by_clf.items():
        lam = _errors_vs_lambda(pooled["y_true"], pooled["y_pred"], pooled["target"])
        lam.insert(0, "classifier", name)
        lam_tables.append(lam)
    lam_df = pd.concat(lam_tables, ignore_index=True)
    lam_df.to_csv(RESULTS_DIR / "passo5_errors_vs_lambda.csv", index=False)

    # Figuras
    # 1) barras métricas
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = summary_a["classifier"].tolist()
    xpos = np.arange(len(names))
    w = 0.25
    ax.bar(xpos - w, summary_a["jaccard_mean"], w, yerr=summary_a["jaccard_std"],
           label="Jaccard", color="#4c72b0", capsize=3)
    ax.bar(xpos, summary_a["f1_micro_mean"], w, label="F1 micro", color="#55a868")
    ax.bar(xpos + w, summary_a["exact_mean"], w, yerr=summary_a["exact_std"],
           label="Exact match", color="#c44e52", capsize=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Passo 5 — modelos afinados (k=4, CV A)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_metrics_tuned.png", dpi=150)
    plt.close(fig)

    # 2) heatmap F1 por FBG
    pivot_f1 = fbg_df.pivot(index="classifier", columns="fbg", values="f1")
    # ordenar linhas como no ranking
    pivot_f1 = pivot_f1.reindex(summary_a["classifier"].tolist())
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot_f1.to_numpy(), aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(13))
    ax.set_xticklabels([str(i) for i in range(13)])
    ax.set_yticks(range(len(pivot_f1)))
    ax.set_yticklabels(pivot_f1.index.tolist())
    ax.set_xlabel("FBG")
    ax.set_title("F1 por canal (dev empilhado, k=4)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_f1_per_fbg_heatmap.png", dpi=150)
    plt.close(fig)

    # 3) erros vs lambda (melhor + MQ baseline)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in [(best_name, "#4c72b0"), ("MQ", "#c44e52")]:
        sub = lam_df[lam_df["classifier"] == name]
        ax.plot(sub["lambda_mean"], sub["jaccard_mean"], "o-", label=name, color=color)
    ax.set_xlabel(r"$\lambda_{res}$ médio do bin (nm)")
    ax.set_ylabel("Jaccard médio")
    ax.set_title("Desempenho vs $\\lambda_{res}$ (bins quantis)")
    ax.legend()
    ax.set_ylim(0.7, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_jaccard_vs_lambda.png", dpi=150)
    plt.close(fig)

    # 4) confusão agregada do melhor (TP/FP/FN por FBG)
    best_fbg = fbg_df[fbg_df["classifier"] == best_name]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x_f = np.arange(13)
    ax.bar(x_f - 0.2, best_fbg["tp"], 0.2, label="TP", color="#55a868")
    ax.bar(x_f, best_fbg["fp"], 0.2, label="FP", color="#c44e52")
    ax.bar(x_f + 0.2, best_fbg["fn"], 0.2, label="FN", color="#ccb974")
    ax.set_xticks(x_f)
    ax.set_xlabel("FBG")
    ax.set_ylabel("Contagem (dev)")
    ax.set_title(f"Erros por canal — {best_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_tp_fp_fn_best.png", dpi=150)
    plt.close(fig)

    # ---------- B) k = 3,4,5 ----------
    print("\n=== B) Comparar k=3,4,5 (X = potências) ===")
    k_rows = []
    for k in (3, 4, 5):
        yk = make_topk_mask(wl, target, k=k)
        assert (yk.sum(axis=1) == k).all()
        for name, est in models.items():
            fold_df, _ = _run_cv(
                name, est, x, yk, target, a_train, a_test, holdout, k=k
            )
            k_rows.append(
                {
                    "k": k,
                    "classifier": name,
                    "jaccard_mean": fold_df["jaccard_samples"].mean(),
                    "jaccard_std": fold_df["jaccard_samples"].std(ddof=0),
                    "exact_mean": fold_df["exact_match"].mean(),
                    "set_recall_mean": fold_df["set_recall"].mean(),
                }
            )
            print(
                f"  k={k} {name:12s} jaccard={k_rows[-1]['jaccard_mean']:.4f}"
            )
    k_df = pd.DataFrame(k_rows)
    k_df.to_csv(RESULTS_DIR / "passo5_compare_k.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name in summary_a["classifier"].tolist():
        sub = k_df[k_df["classifier"] == name]
        ax.plot(sub["k"], sub["jaccard_mean"], "o-", label=name)
    ax.set_xticks([3, 4, 5])
    ax.set_xlabel("k")
    ax.set_ylabel("Jaccard médio")
    ax.set_title("Efeito de k na máscara top-k")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_compare_k.png", dpi=150)
    plt.close(fig)

    # ---------- C) com vs sem posições FBG (k=4) ----------
    print("\n=== C) Com vs sem wl_bragg na entrada (k=4) ===")
    x_pos = np.hstack([x, wl])  # StandardScaler no pipeline trata a escala
    pos_rows = []
    for feat_name, x_use in [("powers_only", x), ("powers_plus_wl", x_pos)]:
        for name, est in models.items():
            # Árvores sem pipeline: wl em nm vs potências ~0-1 — scaler só nos piped.
            # Para RF/AdaBoost, escalar wl manualmente com z-score global do treino em cada fold
            # seria o correto; aqui usamos StandardScaler via wrapper leve.
            if name in ("RandomForest", "AdaBoost"):
                from sklearn.pipeline import Pipeline as SkPipeline
                from sklearn.preprocessing import StandardScaler as SS

                est_use = SkPipeline(
                    [("scaler", SS()), ("clf", copy.deepcopy(est))]
                )
            else:
                est_use = copy.deepcopy(est)
            fold_df, _ = _run_cv(
                name, est_use, x_use, y4, target, a_train, a_test, holdout, k=4
            )
            pos_rows.append(
                {
                    "features": feat_name,
                    "classifier": name,
                    "jaccard_mean": fold_df["jaccard_samples"].mean(),
                    "jaccard_std": fold_df["jaccard_samples"].std(ddof=0),
                    "exact_mean": fold_df["exact_match"].mean(),
                }
            )
            print(
                f"  {feat_name:16s} {name:12s} "
                f"jaccard={pos_rows[-1]['jaccard_mean']:.4f}"
            )
    pos_df = pd.DataFrame(pos_rows)
    pos_df.to_csv(RESULTS_DIR / "passo5_compare_features.csv", index=False)

    # delta table
    piv = pos_df.pivot(index="classifier", columns="features", values="jaccard_mean")
    piv["delta_wl"] = piv["powers_plus_wl"] - piv["powers_only"]
    piv.to_csv(RESULTS_DIR / "passo5_feature_delta.csv")
    print(piv.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = piv.index.tolist()
    xpos = np.arange(len(labels))
    w = 0.35
    ax.bar(xpos - w / 2, piv["powers_only"], w, label="só potências", color="#4c72b0")
    ax.bar(xpos + w / 2, piv["powers_plus_wl"], w, label="+ wl_bragg", color="#55a868")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Jaccard")
    ax.set_title("Entrada: potências vs potências + posições FBG")
    ax.legend(fontsize=8)
    ax.set_ylim(0.85, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo5_compare_features.png", dpi=150)
    plt.close(fig)

    # Meta / achados numéricos para o guia
    best_lam = lam_df[lam_df["classifier"] == best_name]
    edge_j = float(
        min(best_lam.iloc[0]["jaccard_mean"], best_lam.iloc[-1]["jaccard_mean"])
    )
    mid_j = float(best_lam["jaccard_mean"].max())

    meta = {
        "best_classifier_k4": best_name,
        "holdout_used": False,
        "barino_regression_impact": "not_run_optional",
        "summary_k4_ranking": summary_a["classifier"].tolist(),
        "lambda_edge_vs_mid": {
            "classifier": best_name,
            "min_edge_bin_jaccard": edge_j,
            "max_bin_jaccard": mid_j,
        },
        "k_comparison_best_per_k": {
            str(k): k_df[k_df["k"] == k]
            .sort_values("jaccard_mean", ascending=False)
            .iloc[0][["classifier", "jaccard_mean"]]
            .to_dict()
            for k in (3, 4, 5)
        },
        "feature_delta": piv["delta_wl"].round(6).to_dict(),
    }
    with open(RESULTS_DIR / "passo5_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDONE")
    print("Melhor k=4:", best_name, float(summary_a.iloc[0]["jaccard_mean"]))
    print("Δ ao adicionar wl_bragg:", piv["delta_wl"].round(4).to_dict())


if __name__ == "__main__":
    main()
