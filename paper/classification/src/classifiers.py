"""Os 6 classificadores do Passo 3 (defaults documentados; tuning no Passo 4)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from .data_utils import K_DEFAULT, RANDOM_STATE
from .metrics_utils import evaluate_multilabel, extract_scores, topk_from_scores


def _pipe(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def build_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """
    Instâncias novas dos 6 métodos.

    Escala (StandardScaler no treino do fold): kNN, SVM, MLP, MQ.
    Árvores (RF, AdaBoost): sem scaler.
    """
    return {
        "kNN": _pipe(
            KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
        ),
        "SVM": _pipe(
            OneVsRestClassifier(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state),
                n_jobs=-1,
            )
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
        "AdaBoost": OneVsRestClassifier(
            AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=random_state,
            ),
            n_jobs=-1,
        ),
        "MQ": _pipe(MultiOutputRegressor(LinearRegression(), n_jobs=-1)),
    }


def predict_topk_mask(estimator, X: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """Predição alinhada ao problema: top-k FBGs por score."""
    scores = extract_scores(estimator, X)
    return topk_from_scores(scores, k=k)


def fit_predict_fold(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = K_DEFAULT,
) -> dict[str, float]:
    """Treina no fold, prediz top-k e devolve métricas."""
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    # MQ é regressor: treina com alvos 0/1 como contínuos
    estimator.fit(X_train, y_train)
    y_pred = predict_topk_mask(estimator, X_test, k=k)
    metrics = evaluate_multilabel(y_test, y_pred)
    metrics["classifier"] = name  # type: ignore[assignment]
    return metrics


ClassifierFactory = Callable[[], dict[str, Any]]
