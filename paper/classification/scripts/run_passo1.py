"""Executa Passo 1: pré-processamento, checagens e salvamento (dados reais)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import (
    FIGURES_DIR,
    K_DEFAULT,
    RANDOM_STATE,
    RESULTS_DIR,
    WL_RANGE,
    load_prepared_dataset,
    load_measured_dataset,
    prepare_measured_classification,
    save_prepared_dataset,
)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    k = K_DEFAULT

    raw = load_measured_dataset()
    x_raw = np.asarray(raw["input_strength"], dtype=float)
    wl_raw = np.asarray(raw["wl_bragg"], dtype=float)
    t_raw = np.asarray(raw["target"], dtype=float).ravel()
    row_sum_raw = x_raw.sum(axis=1)
    row_min_raw = x_raw.min(axis=1)

    print("=== Bruto ===")
    print(f"n={x_raw.shape[0]}, n_fbgs={x_raw.shape[1]}")
    print(f"lambda_res: min={t_raw.min():.6f}, max={t_raw.max():.6f}")
    print(
        f"soma linha: min={row_sum_raw.min():.6f}, max={row_sum_raw.max():.6f}, "
        f"mean={row_sum_raw.mean():.6f}"
    )
    print(f"min linha: min={row_min_raw.min():.6e}, max={row_min_raw.max():.6e}")
    print(
        f"NaNs: X={np.isnan(x_raw).sum()}, wl={np.isnan(wl_raw).sum()}, "
        f"target={np.isnan(t_raw).sum()}"
    )
    below = int((t_raw <= WL_RANGE[0]).sum())
    above = int((t_raw >= WL_RANGE[1]).sum())
    print(f"fora da faixa: {below + above} (abaixo={below}, acima={above})")

    data = prepare_measured_classification(k=k, wl_range=WL_RANGE)
    x = data["X"]
    y_mask = data["y_mask"]
    wl = data["wl_bragg"]
    target = data["target"]
    keep = data["keep"]

    print("=== Após pipeline ===")
    print(
        f"n_raw={int(data['n_raw'])}, n_kept={int(data['n_kept'])}, "
        f"removidas={int(data['n_raw'] - data['n_kept'])}"
    )
    print(f"X={x.shape}, y_mask={y_mask.shape}")
    print(f"lambda filtrado: min={target.min():.6f}, max={target.max():.6f}")

    checks: dict[str, bool | int] = {}
    row_sum = x.sum(axis=1)
    row_min = x.min(axis=1)
    checks["soma_linhas_~=1"] = bool(np.allclose(row_sum, 1.0, atol=1e-10))
    checks["min_linhas_~=0"] = bool(np.allclose(row_min, 0.0, atol=1e-12))
    checks["X_sem_nan"] = bool(not np.isnan(x).any())
    checks["X_sem_negativos"] = bool((x >= -1e-15).all())
    checks["target_dentro_faixa"] = bool(
        ((target > WL_RANGE[0]) & (target < WL_RANGE[1])).all()
    )
    checks["keep_conta_bate"] = bool(keep.sum() == len(target))
    ones_per_row = y_mask.sum(axis=1)
    checks["exatamente_k_uns"] = bool((ones_per_row == k).all())

    err = np.abs(wl - target.reshape(-1, 1))
    topk_ref = np.argpartition(err, kth=k - 1, axis=1)[:, :k]
    mask_ref = np.zeros_like(y_mask)
    mask_ref[np.arange(len(target)).reshape(-1, 1), topk_ref] = 1
    same_sets = all(
        set(np.flatnonzero(y_mask[i])) == set(np.flatnonzero(mask_ref[i]))
        for i in range(len(target))
    )
    checks["mascara_coincide_topk"] = bool(same_sets)

    topk_sort = np.argsort(err, axis=1)[:, :k]
    mask_sort = np.zeros_like(y_mask)
    mask_sort[np.arange(len(target)).reshape(-1, 1), topk_sort] = 1
    same_sort = all(
        set(np.flatnonzero(y_mask[i])) == set(np.flatnonzero(mask_sort[i]))
        for i in range(len(target))
    )
    checks["mascara_coincide_argsort"] = bool(same_sort)

    kth_err = np.partition(err, k - 1, axis=1)[:, k - 1]
    n_at_kth = (err <= kth_err.reshape(-1, 1) + 1e-15).sum(axis=1)
    n_ties = int((n_at_kth > k).sum())
    checks["amostras_com_empate_no_limiar_k"] = n_ties

    ok_sep = True
    n_sep_fail = 0
    for i in range(len(target)):
        sel = y_mask[i].astype(bool)
        if err[i, sel].max() > err[i, ~sel].min() + 1e-12 and n_at_kth[i] <= k:
            ok_sep = False
            n_sep_fail += 1
    checks["max_sel_leq_min_unsel_sem_empate"] = bool(ok_sep)
    checks["n_sep_fail"] = n_sep_fail

    print("Checagens:")
    for name, ok in checks.items():
        if isinstance(ok, bool):
            print(f"  [{'OK' if ok else 'FALHOU'}] {name}")
        else:
            print(f"  [info] {name} = {ok}")

    bool_checks = {key: val for key, val in checks.items() if isinstance(val, bool)}
    assert all(bool_checks.values()), "Falha de coerência"

    freq = y_mask.mean(axis=0)
    count = y_mask.sum(axis=0)
    bal = pd.DataFrame(
        {
            "fbg_index": np.arange(13),
            "n_positivo": count.astype(int),
            "fracao_positivo": np.round(freq, 4),
            "wl_bragg_media_nm": np.round(wl.mean(axis=0), 3),
        }
    )
    print(f"Esperado uniforme k/13={k / 13:.4f}")
    print(f"std frações={freq.std(ddof=0):.4f}")
    print(bal.to_string(index=False))
    print("wl_bragg std por FBG:", np.round(wl.std(axis=0), 4))
    print("wl_bragg min/max global:", float(wl.min()), float(wl.max()))

    mean_res_when_pos = []
    for j in range(13):
        sel = y_mask[:, j] == 1
        mean_res_when_pos.append(target[sel].mean() if sel.any() else np.nan)
    mean_res_when_pos = np.array(mean_res_when_pos)
    wl_mean = wl.mean(axis=0)
    corr = float(np.corrcoef(wl_mean, mean_res_when_pos)[0, 1])
    print("corr(wl_mean, mean_res_when_pos)=", corr)
    print("diff mean_res - wl_mean:", np.round(mean_res_when_pos - wl_mean, 3))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(np.arange(13), freq, color="#4c72b0", edgecolor="black", linewidth=0.4)
    axes[0].axhline(k / 13, color="C3", ls="--", label=f"uniforme k/13={k / 13:.3f}")
    axes[0].set_xlabel("Índice do FBG")
    axes[0].set_ylabel("Fração com rótulo 1")
    axes[0].set_title(f"Frequência da máscara top-{k}")
    axes[0].legend(fontsize=8)
    axes[1].hist(target, bins=40, color="#55a868", edgecolor="white")
    axes[1].set_xlabel(r"$\lambda_{res}$ (nm)")
    axes[1].set_ylabel("Contagem")
    axes[1].set_title(r"Distribuição de $\lambda_{res}$ (filtrado)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo1_balance_mask.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_mean, mean_res_when_pos, "o-", color="#4c72b0")
    lims = [
        min(wl_mean.min(), np.nanmin(mean_res_when_pos)) - 2,
        max(wl_mean.max(), np.nanmax(mean_res_when_pos)) + 2,
    ]
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    for j in range(13):
        ax.text(wl_mean[j], mean_res_when_pos[j], str(j), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Posição média do FBG (nm)")
    ax.set_ylabel(r"Média de $\lambda_{res}$ quando FBG=1")
    ax.set_title("Coerência espacial da máscara")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo1_mask_vs_lambda.png", dpi=150)
    plt.close(fig)

    out_npz = save_prepared_dataset(data)
    print(f"Salvo: {out_npz} ({out_npz.stat().st_size / 1024:.1f} KB)")

    meta = {
        "source": "paper/fbg-demodulated-lpfg/data/measured.dataset",
        "n_raw": int(data["n_raw"]),
        "n_kept": int(data["n_kept"]),
        "n_removed": int(data["n_raw"] - data["n_kept"]),
        "n_fbgs": 13,
        "k": int(k),
        "wl_range_nm": list(WL_RANGE),
        "random_state": int(RANDOM_STATE),
        "X": "input_strength normalizado (min-subtract + soma=1)",
        "y_mask": f"multi-rótulo top-{k} por |wl_bragg - target|",
        "mask_positive_fraction_per_fbg": [float(f) for f in freq],
        "n_samples_with_tie_at_kth": n_ties,
        "coherence_checks_passed": True,
        "corr_wl_mean_vs_mean_res_when_pos": corr,
    }
    meta_path = RESULTS_DIR / "prepared_measured_k4_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    bal.to_csv(RESULTS_DIR / "passo1_mask_balance.csv", index=False)
    print(f"Salvo: {meta_path}")
    print(f"Salvo: {RESULTS_DIR / 'passo1_mask_balance.csv'}")

    reloaded = load_prepared_dataset(out_npz)
    assert reloaded["X"].shape == x.shape
    assert np.array_equal(reloaded["y_mask"], y_mask)
    print("Reload OK")
    print("DONE")


if __name__ == "__main__":
    main()
