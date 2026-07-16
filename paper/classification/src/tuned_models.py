"""Classificadores com hiperparâmetros de consenso do Passo 4 (nested CV)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import RANDOM_STATE


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def build_tuned_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """
    Consenso / moda dos best_params do Passo 4 multiclasse.

    Valores iniciais razoáveis; atualizar após Passo 4 se o consenso mudar.
    """
    return {
        "kNN": _pipe(
            KNeighborsClassifier(
                n_neighbors=11, weights="distance", metric="minkowski", p=2
            )
        ),
        "SVM": _pipe(
            SVC(kernel="rbf", C=10.0, gamma=0.1, random_state=random_state)
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
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=random_state,
        ),
        "MQ": _pipe(RidgeClassifier(alpha=1.0, random_state=random_state)),
    }
