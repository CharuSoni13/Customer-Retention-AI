"""Model evaluation metrics + cross validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = y_pred
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def compare_models(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in fitted_models.items():
        metrics = evaluate(model, X_test, y_test)
        metrics.pop("confusion_matrix", None)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def cv_score(model, X, y, n_splits=5, scoring="roc_auc", random_state=42) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}
