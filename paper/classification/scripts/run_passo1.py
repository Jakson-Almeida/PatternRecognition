"""Executa Passo 1: pré-processamento, rótulos multiclasse e salvamento."""

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
    N_CLASSES,
    N_FBGS,
    RANDOM_STATE,
    RESULTS_DIR,
    WL_RANGE,
    class_to_mask,
    load_prepared_dataset,
    load_measured_dataset,
    mask_to_window_class,
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
    y_class = np.asarray(data["y_class"], dtype=int).ravel()
    wl = data["wl_bragg"]
    target = data["target"]
    keep = data["keep"]
    n_classes = int(data["n_classes"])

    print("=== Após pipeline ===")
    print(
        f"n_raw={int(data['n_raw'])}, n_kept={int(data['n_kept'])}, "
        f"removidas={int(data['n_raw'] - data['n_kept'])}"
    )
    print(f"X={x.shape}, y_mask={y_mask.shape}, y_class={y_class.shape}")
    print(f"n_classes={n_classes} (esperado {N_CLASSES})")
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

    y_from_mask = mask_to_window_class(y_mask, k=k)
    checks["y_class_bate_com_mascara"] = bool(np.array_equal(y_class, y_from_mask))
    checks["bijecao_classe_mascara"] = bool(
        np.array_equal(class_to_mask(y_class, n_fbgs=N_FBGS, k=k), y_mask)
    )
    checks["n_classes_esperado"] = bool(n_classes == N_CLASSES)
    checks["classes_em_0_a_9"] = bool(y_class.min() == 0 and y_class.max() == N_CLASSES - 1)
    checks["todas_as_10_classes_presentes"] = bool(len(np.unique(y_class)) == N_CLASSES)

    print("Checagens:")
    for name, ok in checks.items():
        if isinstance(ok, bool):
            print(f"  [{'OK' if ok else 'FALHOU'}] {name}")
        else:
            print(f"  [info] {name} = {ok}")

    bool_checks = {key: val for key, val in checks.items() if isinstance(val, bool)}
    assert all(bool_checks.values()), "Falha de coerência"

    counts = np.bincount(y_class, minlength=N_CLASSES)
    class_bal = pd.DataFrame(
        {
            "class": np.arange(N_CLASSES),
            "window": [f"{{{','.join(map(str, range(s, s + k)))}}}" for s in range(N_CLASSES)],
            "n": counts.astype(int),
            "frac": np.round(counts / len(y_class), 4),
        }
    )
    print(class_bal.to_string(index=False))
    assert int(counts.sum()) == len(y_class)

    # balanceamento por FBG (máscara derivada) — diagnóstico
    freq = y_mask.mean(axis=0)
    bal_fbg = pd.DataFrame(
        {
            "fbg_index": np.arange(N_FBGS),
            "n_positivo": y_mask.sum(axis=0).astype(int),
            "fracao_positivo": np.round(freq, 4),
            "wl_bragg_media_nm": np.round(wl.mean(axis=0), 3),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(np.arange(N_CLASSES), counts, color="#4c72b0", edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("Classe (início da janela)")
    axes[0].set_ylabel("Contagem")
    axes[0].set_title(f"Distribuição das {N_CLASSES} classes")
    axes[0].set_xticks(np.arange(N_CLASSES))
    axes[1].hist(target, bins=40, color="#55a868", edgecolor="white")
    axes[1].set_xlabel(r"$\lambda_{res}$ (nm)")
    axes[1].set_ylabel("Contagem")
    axes[1].set_title(r"Distribuição de $\lambda_{res}$ (filtrado)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo1_balance_mask.png", dpi=150)
    plt.close(fig)

    mean_res_when_pos = []
    for j in range(N_FBGS):
        sel = y_mask[:, j] == 1
        mean_res_when_pos.append(target[sel].mean() if sel.any() else np.nan)
    mean_res_when_pos = np.array(mean_res_when_pos)
    wl_mean = wl.mean(axis=0)
    corr = float(np.corrcoef(wl_mean, mean_res_when_pos)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_mean, mean_res_when_pos, "o-", color="#4c72b0")
    lims = [
        min(wl_mean.min(), np.nanmin(mean_res_when_pos)) - 2,
        max(wl_mean.max(), np.nanmax(mean_res_when_pos)) + 2,
    ]
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    for j in range(N_FBGS):
        ax.text(wl_mean[j], mean_res_when_pos[j], str(j), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Posição média do FBG (nm)")
    ax.set_ylabel(r"Média de $\lambda_{res}$ quando FBG na janela")
    ax.set_title("Coerência espacial da janela (máscara)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "passo1_mask_vs_lambda.png", dpi=150)
    plt.close(fig)

    out_npz = save_prepared_dataset(data)
    print(f"Salvo: {out_npz} ({out_npz.stat().st_size / 1024:.1f} KB)")

    meta = {
        "source": "paper/fbg-demodulated-lpfg/data/measured.dataset",
        "formulation": "multiclass_contiguous_window",
        "n_raw": int(data["n_raw"]),
        "n_kept": int(data["n_kept"]),
        "n_removed": int(data["n_raw"] - data["n_kept"]),
        "n_fbgs": N_FBGS,
        "k": int(k),
        "n_classes": N_CLASSES,
        "wl_range_nm": list(WL_RANGE),
        "random_state": int(RANDOM_STATE),
        "X": "input_strength normalizado (min-subtract + soma=1)",
        "y_class": f"classe s em 0..{N_CLASSES - 1}; janela FBGs {{s..s+{k - 1}}}",
        "y_mask": f"máscara equivalente top-{k} / janela contígua",
        "class_counts": [int(c) for c in counts],
        "coherence_checks_passed": True,
        "corr_wl_mean_vs_mean_res_when_pos": corr,
    }
    meta_path = RESULTS_DIR / "prepared_measured_k4_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    class_bal.to_csv(RESULTS_DIR / "passo1_class_balance.csv", index=False)
    bal_fbg.to_csv(RESULTS_DIR / "passo1_mask_balance.csv", index=False)
    print(f"Salvo: {meta_path}")
    print(f"Salvo: {RESULTS_DIR / 'passo1_class_balance.csv'}")

    reloaded = load_prepared_dataset(out_npz)
    assert reloaded["X"].shape == x.shape
    assert np.array_equal(reloaded["y_mask"], y_mask)
    assert np.array_equal(reloaded["y_class"], y_class)
    print("Reload OK")
    print("DONE")


if __name__ == "__main__":
    main()
