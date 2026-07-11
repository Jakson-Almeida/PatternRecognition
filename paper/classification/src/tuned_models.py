"""Classificadores com hiperparâmetros de consenso do Passo 4 (nested CV)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from .data_utils import RANDOM_STATE


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def build_tuned_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """
    Consenso estável / moda dos best_params do Passo 4.

    SVM: C=10, gamma=0.1 em todos os folds.
    Demais: escolha mais frequente ou configuração documentada no guia.
    """
    return {
        "kNN": _pipe(
            KNeighborsClassifier(
                n_neighbors=11, weights="distance", metric="minkowski", p=2
            )
        ),
        "SVM": _pipe(
            OneVsRestClassifier(
                SVC(kernel="rbf", C=10.0, gamma=0.1, random_state=random_state),
                n_jobs=-1,
            )
        ),
        "MLP": _pipe(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                activation="relu",
                solver="adam",
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state,
            )
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        ),
        "AdaBoost": OneVsRestClassifier(
            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=random_state,
            ),
            n_jobs=-1,
        ),
        "MQ": _pipe(MultiOutputRegressor(Ridge(alpha=0.01), n_jobs=-1)),
    }
