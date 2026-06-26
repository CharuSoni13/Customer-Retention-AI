"""Train and compare multiple churn classifiers."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE


def _try_import_boosters() -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        from xgboost import XGBClassifier
        out["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=-1,
            num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        out["CatBoost"] = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6,
            random_state=RANDOM_STATE, verbose=False,
        )
    except Exception:
        pass
    return out


def build_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    models.update(_try_import_boosters())
    return models


def fit_all(models: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            fitted[name] = model
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {exc}")
    return fitted
