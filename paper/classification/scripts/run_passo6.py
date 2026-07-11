"""Passo 6: figuras prontas para o artigo IEEE (dados reais)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifiers import predict_topk_mask
from src.cv_utils import load_cv_splits
from src.data_utils import (
    FIGURES_DIR,
    K_DEFAULT,
    PAPER_ROOT,
    RESULTS_DIR,
    load_prepared_dataset,
)
from src.tuned_models import build_tuned_classifiers

IEEE_FIG = PAPER_ROOT / "ieee" / "figures"
CLASS_FIG = FIGURES_DIR


def _as_index_array(obj) -> np.ndarray:
    return np.asarray(obj, dtype=int).ravel()


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
        }
    )


def fig_schema_lpg_fbg(out: Path) -> None:
    """Esquema: vale LPFG + 13 FBGs + máscara top-4 (ilustrativo, posições típicas)."""
    # Posições médias reais do dataset preparado
    data = load_prepared_dataset()
    wl_mean = np.asarray(data["wl_bragg"], dtype=float).mean(axis=0)
    # lambda_res ilustrativa no centro do array
    lam = float(np.median(data["target"]))
    err = np.abs(wl_mean - lam)
    top4 = set(np.argsort(err)[:K_DEFAULT].tolist())

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    # curva LPFG esquemática (Lorentziana invertida)
    xs = np.linspace(wl_mean.min() - 5, wl_mean.max() + 5, 500)
    dip = 1.0 / (1.0 + ((xs - lam) / 4.5) ** 2)
    ax.plot(xs, 1 - 0.85 * dip, color="#2c3e50", lw=1.8, label="LPFG (esquema)")
    ax.axvline(lam, color="#c44e52", ls="--", lw=1.2, label=r"$\lambda_{res}$")

    for i, w in enumerate(wl_mean):
        selected = i in top4
        ax.plot(
            [w, w],
            [0.05, 0.95],
            color="#4c72b0" if selected else "#bbbbbb",
            lw=2.2 if selected else 1.0,
            solid_capstyle="round",
        )
        ax.plot(
            w,
            0.08,
            marker="o",
            markersize=7 if selected else 5,
            color="#4c72b0" if selected else "#888888",
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
        ax.text(w, 1.02, str(i), ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Transmitância (u.a.)")
    ax.set_title("Esquema: LPFG, array de 13 FBGs e máscara top-4")
    # legenda manual
    ax.plot([], [], color="#4c72b0", lw=2.2, label="FBG na máscara (top-4)")
    ax.plot([], [], color="#bbbbbb", lw=1.0, label="FBG fora da máscara")
    ax.legend(loc="lower right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_example_true_vs_pred(out: Path) -> dict:
    """Exemplo real: potências FBG com máscara verdadeira vs SVM prevista."""
    data = load_prepared_dataset()
    x = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y_mask"], dtype=int)
    wl = np.asarray(data["wl_bragg"], dtype=float)
    target = np.asarray(data["target"], dtype=float).ravel()
    splits = load_cv_splits()
    tr = _as_index_array(splits["a_train"][0])
    te = _as_index_array(splits["a_test"][0])

    clf = build_tuned_classifiers()["SVM"]
    clf.fit(x[tr], y[tr])
    pred = predict_topk_mask(clf, x[te], k=K_DEFAULT)

    # escolher 1 acerto exato e 1 erro (se existir)
    exact = (pred == y[te]).all(axis=1)
    idx_ok = int(np.flatnonzero(exact)[0])
    idx_bad = int(np.flatnonzero(~exact)[0]) if (~exact).any() else idx_ok

    samples = [("acerto", te[idx_ok], y[te][idx_ok], pred[idx_ok]),
               ("erro", te[idx_bad], y[te][idx_bad], pred[idx_bad])]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for ax, (tag, glob_i, yt, yp) in zip(axes, samples):
        powers = x[glob_i]
        wls = wl[glob_i]
        lam = target[glob_i]
        xpos = np.arange(13)
        ax.bar(xpos, powers, color="#d0d0d0", edgecolor="white", width=0.7, label="potência")
        for j in range(13):
            if yt[j] == 1:
                ax.plot(j, powers[j] + 0.01, "v", color="#4c72b0", ms=8, label="verdadeiro" if j == np.flatnonzero(yt)[0] else None)
            if yp[j] == 1:
                ax.plot(j, powers[j] + 0.035, "^", color="#c44e52", ms=8, label="previsto" if j == np.flatnonzero(yp)[0] else None)
        ax.axvline(np.interp(lam, wls, xpos.astype(float)), color="#2ca02c", ls=":", lw=1.2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{wls[j]:.0f}" for j in range(13)], rotation=45, fontsize=7)
        ax.set_xlabel(r"$\lambda_{\mathrm{FBG}}$ (nm)")
        ax.set_title(f"{tag.capitalize()}  |  " + r"$\lambda_{res}$=" + f"{lam:.1f} nm")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Potência normalizada")
    fig.suptitle("Máscara top-4: verdadeira (▼) vs SVM (▲)", y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    return {
        "fold": 0,
        "classifier": "SVM",
        "sample_ok_global_idx": int(te[idx_ok]),
        "sample_err_global_idx": int(te[idx_bad]),
        "n_exact_in_fold_test": int(exact.sum()),
        "n_test": int(len(te)),
    }


def fig_metrics_bars(out: Path) -> None:
    summary = pd.read_csv(RESULTS_DIR / "passo5_summary_k4.csv").sort_values(
        "jaccard_mean", ascending=False
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = summary["classifier"].tolist()
    xpos = np.arange(len(names))
    w = 0.25
    ax.bar(
        xpos - w,
        summary["jaccard_mean"],
        w,
        yerr=summary["jaccard_std"],
        label="Jaccard",
        color="#4c72b0",
        capsize=3,
    )
    ax.bar(xpos, summary["f1_micro_mean"], w, label="F1 micro", color="#55a868")
    ax.bar(
        xpos + w,
        summary["exact_mean"],
        w,
        yerr=summary["exact_std"],
        label="Exact match",
        color="#c44e52",
        capsize=3,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.78, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Comparação dos classificadores (CV, k=4)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_error_heatmap(out: Path) -> None:
    fbg = pd.read_csv(RESULTS_DIR / "passo5_per_fbg_metrics.csv")
    # taxa de erro por canal = (FP+FN)/n ; n = tp+fp+fn+tn
    fbg["n"] = fbg["tp"] + fbg["fp"] + fbg["fn"] + fbg["tn"]
    fbg["err"] = (fbg["fp"] + fbg["fn"]) / fbg["n"]
    order = (
        pd.read_csv(RESULTS_DIR / "passo5_summary_k4.csv")
        .sort_values("jaccard_mean", ascending=False)["classifier"]
        .tolist()
    )
    mat = fbg.pivot(index="classifier", columns="fbg", values="err").reindex(order)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.12)
    ax.set_xticks(range(13))
    ax.set_xticklabels([str(i) for i in range(13)])
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index.tolist())
    ax.set_xlabel("Índice do FBG")
    ax.set_title("Taxa de erro por canal (FP+FN)/N")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Erro")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_jaccard_vs_lambda(out: Path) -> None:
    """Reusa análise do Passo 5 com estilo de artigo (SVM + MQ)."""
    lam = pd.read_csv(RESULTS_DIR / "passo5_errors_vs_lambda.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for name, color in [("SVM", "#4c72b0"), ("MQ", "#c44e52"), ("RandomForest", "#55a868")]:
        sub = lam[lam["classifier"] == name]
        ax.plot(sub["lambda_mean"], sub["jaccard_mean"], "o-", color=color, label=name, ms=4)
    ax.set_xlabel(r"$\lambda_{res}$ médio do bin (nm)")
    ax.set_ylabel("Jaccard médio")
    ax.set_title(r"Desempenho ao longo de $\lambda_{res}$")
    ax.set_ylim(0.75, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    _style()
    IEEE_FIG.mkdir(parents=True, exist_ok=True)
    CLASS_FIG.mkdir(parents=True, exist_ok=True)

    outputs = {
        "schema": IEEE_FIG / "fig_schema_lpg_fbg.png",
        "example": IEEE_FIG / "fig_mask_true_vs_pred.png",
        "metrics": IEEE_FIG / "fig_metrics_classifiers.png",
        "heatmap": IEEE_FIG / "fig_error_heatmap_fbg.png",
        "lambda": IEEE_FIG / "fig_jaccard_vs_lambda.png",
    }

    print("Gerando esquema...")
    fig_schema_lpg_fbg(outputs["schema"])
    print("Gerando exemplo true vs pred...")
    meta_ex = fig_example_true_vs_pred(outputs["example"])
    print("Gerando barras de métricas...")
    fig_metrics_bars(outputs["metrics"])
    print("Gerando heatmap de erros...")
    fig_error_heatmap(outputs["heatmap"])
    print("Gerando Jaccard vs lambda...")
    fig_jaccard_vs_lambda(outputs["lambda"])

    # cópias também em classification/figures
    for key, path in outputs.items():
        shutil.copy2(path, CLASS_FIG / path.name)

    # opcional ROC: não gerado (deixado para se necessário no artigo)
    meta = {
        "ieee_figures_dir": str(IEEE_FIG.relative_to(PAPER_ROOT.parent)).replace("\\", "/"),
        "figures": {k: v.name for k, v in outputs.items()},
        "example_meta": meta_ex,
        "roc_pr_curves": "not_generated_optional",
        "notes": [
            "Esquema usa posições médias reais de wl_bragg e lambda_res mediana.",
            "Exemplo true/pred: SVM afinado, fold 0 da estratégia A (hold-out intacto).",
            "Métricas/heatmap/lambda: artefatos reais do Passo 5.",
        ],
    }
    meta_path = RESULTS_DIR / "passo6_figures_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nFiguras salvas em:", IEEE_FIG)
    for k, p in outputs.items():
        print(f"  [{k}] {p.name} ({p.stat().st_size/1024:.1f} KB)")
    print("Meta:", meta_path)
    print("DONE")


if __name__ == "__main__":
    main()
