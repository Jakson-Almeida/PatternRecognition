"""Utilitários de validação cruzada e hold-out (Passo 2) — multiclasse."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from .data_utils import N_CLASSES, RANDOM_STATE, RESULTS_DIR

# Defaults documentados — não alterar sem regenerar os splits salvos.
HOLD_OUT_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 5
N_LAMBDA_BINS = 10


@dataclass(frozen=True)
class SplitConfig:
    hold_out_size: float = HOLD_OUT_SIZE
    n_splits: int = N_SPLITS
    n_repeats: int = N_REPEATS
    n_lambda_bins: int = N_LAMBDA_BINS
    random_state: int = RANDOM_STATE
    stratify_by: str = "y_class"


def make_lambda_bins(target: np.ndarray, n_bins: int = N_LAMBDA_BINS) -> np.ndarray:
    """Bins por quantis de lambda_res (diagnóstico de cobertura espectral)."""
    target = np.asarray(target, dtype=float).ravel()
    labels = pd.qcut(target, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(labels, dtype=int)


def make_holdout_split(
    n_samples: int,
    stratify_labels: np.ndarray,
    test_size: float = HOLD_OUT_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Separa hold-out final (só relatório). Retorna (dev_idx, holdout_idx)."""
    indices = np.arange(n_samples)
    stratify_labels = np.asarray(stratify_labels).ravel()
    if len(stratify_labels) != n_samples:
        raise ValueError("stratify_labels deve ter o mesmo comprimento que n_samples")
    dev_idx, holdout_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return np.sort(dev_idx), np.sort(holdout_idx)


def iter_strategy_a(
    stratify_labels: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Estratégia A: StratifiedKFold (uma passagem)."""
    y = np.asarray(stratify_labels).ravel()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        yield fold, tr, te


def iter_strategy_b(
    stratify_labels: np.ndarray,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> Iterator[tuple[int, int, int, np.ndarray, np.ndarray]]:
    """Estratégia B: RepeatedStratifiedKFold. Yield (rep, fold, global_id, tr, te)."""
    y = np.asarray(stratify_labels).ravel()
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    for global_id, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        rep = global_id // n_splits
        fold = global_id % n_splits
        yield rep, fold, global_id, tr, te


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    inter = np.intersect1d(train_idx, test_idx)
    if len(inter) > 0:
        raise AssertionError(f"Vazamento: {len(inter)} índices em treino e teste")


def fold_balance_report(
    name: str,
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target: np.ndarray,
    y_class: np.ndarray,
    stratify_labels: np.ndarray,
    repeat: int | None = None,
) -> dict:
    """Estatísticas de coerência de um fold."""
    assert_no_leakage(train_idx, test_idx)
    target = np.asarray(target, dtype=float).ravel()
    y_class = np.asarray(y_class, dtype=int).ravel()
    stratify_labels = np.asarray(stratify_labels).ravel()

    def _part(idx: np.ndarray) -> dict:
        counts = np.bincount(y_class[idx], minlength=N_CLASSES)
        return {
            "n": int(len(idx)),
            "lambda_min": float(target[idx].min()),
            "lambda_max": float(target[idx].max()),
            "lambda_mean": float(target[idx].mean()),
            "lambda_std": float(target[idx].std(ddof=0)),
            "strat_bin_counts": {
                str(int(b)): int(c)
                for b, c in zip(*np.unique(stratify_labels[idx], return_counts=True))
            },
            "class_counts": [int(x) for x in counts],
            "class_frac": [float(x) for x in counts / max(len(idx), 1)],
        }

    out = {
        "strategy": name,
        "fold": int(fold_id),
        "train": _part(train_idx),
        "test": _part(test_idx),
    }
    if repeat is not None:
        out["repeat"] = int(repeat)
    return out


def summarize_strategy_balance(reports: list[dict]) -> pd.DataFrame:
    """Tabela resumo: tamanho e contagens por classe no teste de cada fold."""
    rows = []
    for r in reports:
        row = {
            "strategy": r["strategy"],
            "repeat": r.get("repeat", 0),
            "fold": r["fold"],
            "n_train": r["train"]["n"],
            "n_test": r["test"]["n"],
            "lambda_mean_train": r["train"]["lambda_mean"],
            "lambda_mean_test": r["test"]["lambda_mean"],
        }
        for j, c in enumerate(r["test"]["class_counts"]):
            row[f"test_class_{j}"] = c
        rows.append(row)
    return pd.DataFrame(rows)


def build_and_save_splits(
    target: np.ndarray,
    y_class: np.ndarray,
    config: SplitConfig | None = None,
    out_dir: Path | None = None,
) -> dict:
    """
    Constrói hold-out + estratégias A e B estratificados por y_class.

    Índices relativos ao dataset preparado completo (7300 amostras).
    """
    config = config or SplitConfig()
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target = np.asarray(target, dtype=float).ravel()
    y_class = np.asarray(y_class, dtype=int).ravel()
    n = len(target)
    if len(y_class) != n:
        raise ValueError("y_class e target com comprimentos diferentes")

    if config.stratify_by != "y_class":
        raise ValueError(
            f"stratify_by={config.stratify_by!r} não suportado; use 'y_class'."
        )

    all_bins = make_lambda_bins(target, n_bins=config.n_lambda_bins)

    dev_idx, holdout_idx = make_holdout_split(
        n, y_class, test_size=config.hold_out_size, random_state=config.random_state
    )
    assert_no_leakage(dev_idx, holdout_idx)
    if len(np.union1d(dev_idx, holdout_idx)) != n:
        raise AssertionError("Hold-out + dev não cobrem todas as amostras")

    dev_y = y_class[dev_idx]
    holdout_y = y_class[holdout_idx]

    folds_a_train: list[np.ndarray] = []
    folds_a_test: list[np.ndarray] = []
    reports_a: list[dict] = []
    for fold, tr_rel, te_rel in iter_strategy_a(
        dev_y, n_splits=config.n_splits, random_state=config.random_state
    ):
        tr = dev_idx[tr_rel]
        te = dev_idx[te_rel]
        assert_no_leakage(tr, te)
        if np.intersect1d(tr, holdout_idx).size or np.intersect1d(te, holdout_idx).size:
            raise AssertionError("Fold A intersecta hold-out")
        folds_a_train.append(tr)
        folds_a_test.append(te)
        reports_a.append(
            fold_balance_report(
                "A_StratifiedKFold",
                fold,
                tr,
                te,
                target,
                y_class,
                y_class,
            )
        )

    folds_b_train: list[np.ndarray] = []
    folds_b_test: list[np.ndarray] = []
    folds_b_repeat: list[int] = []
    folds_b_fold: list[int] = []
    reports_b: list[dict] = []
    for rep, fold, gid, tr_rel, te_rel in iter_strategy_b(
        dev_y,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    ):
        tr = dev_idx[tr_rel]
        te = dev_idx[te_rel]
        assert_no_leakage(tr, te)
        if np.intersect1d(tr, holdout_idx).size or np.intersect1d(te, holdout_idx).size:
            raise AssertionError("Fold B intersecta hold-out")
        folds_b_train.append(tr)
        folds_b_test.append(te)
        folds_b_repeat.append(rep)
        folds_b_fold.append(fold)
        reports_b.append(
            fold_balance_report(
                "B_RepeatedStratifiedKFold",
                fold,
                tr,
                te,
                target,
                y_class,
                y_class,
                repeat=rep,
            )
        )

    summary_a = summarize_strategy_balance(reports_a)
    summary_b = summarize_strategy_balance(reports_b)

    class_cols = [c for c in summary_a.columns if c.startswith("test_class_")]
    min_class_a = int(summary_a[class_cols].min().min())
    min_class_b = int(summary_b[class_cols].min().min())

    meta = {
        "config": asdict(config),
        "n_total": int(n),
        "n_dev": int(len(dev_idx)),
        "n_holdout": int(len(holdout_idx)),
        "n_classes": int(N_CLASSES),
        "n_lambda_bins_effective": int(len(np.unique(all_bins))),
        "holdout_class_counts": {
            str(int(b)): int(c) for b, c in zip(*np.unique(holdout_y, return_counts=True))
        },
        "dev_class_counts": {
            str(int(b)): int(c) for b, c in zip(*np.unique(dev_y, return_counts=True))
        },
        "strategy_a": {
            "name": "StratifiedKFold",
            "n_folds": int(config.n_splits),
            "min_test_count_any_class": min_class_a,
        },
        "strategy_b": {
            "name": "RepeatedStratifiedKFold",
            "n_splits": int(config.n_splits),
            "n_repeats": int(config.n_repeats),
            "n_evaluations": int(config.n_splits * config.n_repeats),
            "min_test_count_any_class": min_class_b,
        },
        "notes": [
            "Estratificação por y_class (10 janelas contíguas).",
            "Hold-out isolado: não entra em nenhum fold de A ou B.",
            "lambda_bins salvos só para diagnóstico de cobertura espectral.",
            "Índices relativos ao prepared_measured_k4.npz.",
        ],
        "coherence_ok": True,
    }

    splits_path = out_dir / "cv_splits_passo2.npz"
    np.savez_compressed(
        splits_path,
        dev_idx=dev_idx,
        holdout_idx=holdout_idx,
        y_class=y_class,
        lambda_bins=all_bins,
        a_train=np.array(folds_a_train, dtype=object),
        a_test=np.array(folds_a_test, dtype=object),
        b_train=np.array(folds_b_train, dtype=object),
        b_test=np.array(folds_b_test, dtype=object),
        b_repeat=np.asarray(folds_b_repeat, dtype=int),
        b_fold=np.asarray(folds_b_fold, dtype=int),
        config_hold_out_size=np.float64(config.hold_out_size),
        config_n_splits=np.int64(config.n_splits),
        config_n_repeats=np.int64(config.n_repeats),
        config_n_lambda_bins=np.int64(config.n_lambda_bins),
        config_random_state=np.int64(config.random_state),
        config_stratify_by=np.asarray(config.stratify_by),
    )

    meta_path = out_dir / "cv_splits_passo2_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    summary_a.to_csv(out_dir / "passo2_strategy_a_balance.csv", index=False)
    summary_b.to_csv(out_dir / "passo2_strategy_b_balance.csv", index=False)

    with open(out_dir / "passo2_fold_reports_a.json", "w", encoding="utf-8") as f:
        json.dump(reports_a, f, indent=2)
    with open(out_dir / "passo2_fold_reports_b.json", "w", encoding="utf-8") as f:
        json.dump(reports_b, f, indent=2)

    return {
        "meta": meta,
        "dev_idx": dev_idx,
        "holdout_idx": holdout_idx,
        "y_class": y_class,
        "lambda_bins": all_bins,
        "reports_a": reports_a,
        "reports_b": reports_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "splits_path": splits_path,
        "meta_path": meta_path,
    }


def load_cv_splits(path: Path | None = None) -> dict:
    """Carrega artefato de splits do Passo 2."""
    path = path or (RESULTS_DIR / "cv_splits_passo2.npz")
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}
