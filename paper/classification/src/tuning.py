"""Ajuste de hiperparâmetros com nested CV (Passo 4) — multiclasse."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_utils import RANDOM_STATE
from .metrics_utils import evaluate_multiclass


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def build_search_spaces(random_state: int = RANDOM_STATE) -> dict[str, dict[str, Any]]:
    return {
        "kNN": {
            "estimator": _pipe(KNeighborsClassifier(metric="minkowski", p=2)),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"],
            },
        },
        "SVM": {
            "estimator": _pipe(SVC(kernel="rbf", random_state=random_state)),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", 0.1],
            },
        },
        "MLP": {
            "estimator": _pipe(
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=random_state,
                )
            ),
            "param_grid": {
                "clf__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
                "clf__alpha": [1e-4, 1e-3],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(n_jobs=1, random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2],
            },
        },
        "AdaBoost": {
            "estimator": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                ),
                random_state=random_state,
            ),
            "param_grid": {
                "estimator__max_depth": [2, 3, 4],
                "n_estimators": [100, 200],
                "learning_rate": [0.3, 0.5, 1.0],
            },
        },
        "MQ": {
            "estimator": _pipe(RidgeClassifier(random_state=random_state)),
            "param_grid": {
                "clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
    }


def nested_tune_fold(
    name: str,
    estimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    stratify_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    inner_splits: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Inner GridSearchCV (accuracy); avalia no teste externo."""
    y_train = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()
    stratify_train = np.asarray(stratify_train).ravel()

    inner = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state
    )
    inner_cv = list(inner.split(np.zeros(len(y_train)), stratify_train))

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    y_pred = np.asarray(search.best_estimator_.predict(X_test), dtype=int).ravel()
    metrics = evaluate_multiclass(y_test, y_pred)
    return {
        "classifier": name,
        "best_params": search.best_params_,
        "best_inner_accuracy": float(search.best_score_),
        "metrics": metrics,
        "n_candidates": int(len(search.cv_results_["params"])),
    }
