"""Os 6 classificadores multiclasse (defaults; tuning no Passo 4)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_utils import RANDOM_STATE
from .metrics_utils import evaluate_multiclass


def _pipe(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def build_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """
    Instâncias novas dos 6 métodos multiclasse.

    Escala: kNN, SVM, MLP, MQ (RidgeClassifier).
    Árvores (RF, AdaBoost): sem scaler.
    """
    return {
        "kNN": _pipe(
            KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
        ),
        "SVM": _pipe(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
        ),
        "MLP": _pipe(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
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
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3,
                class_weight="balanced",
                random_state=random_state,
            ),
            n_estimators=200,
            learning_rate=0.5,
            random_state=random_state,
        ),
        "MQ": _pipe(RidgeClassifier(alpha=1.0, random_state=random_state)),
    }


def fit_predict_fold(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Treina no fold, prediz classes e devolve métricas."""
    y_train = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    estimator.fit(X_train, y_train)
    y_pred = np.asarray(estimator.predict(X_test), dtype=int).ravel()
    metrics = evaluate_multiclass(y_test, y_pred)
    metrics["classifier"] = name  # type: ignore[assignment]
    return metrics


ClassifierFactory = Callable[[], dict[str, Any]]
