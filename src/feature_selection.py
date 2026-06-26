"""Compare feature selection techniques."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression


def random_forest_importance(X, y, feature_names, n_estimators=200, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)


def mutual_information(X, y, feature_names, random_state=42):
    mi = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi, index=feature_names).sort_values(ascending=False)


def rfe_ranking(X, y, feature_names, n_features=15):
    estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    return pd.Series(rfe.ranking_, index=feature_names).sort_values()


def select_k_best(X, y, feature_names, k=15):
    skb = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    skb.fit(X, y)
    return pd.Series(skb.scores_, index=feature_names).sort_values(ascending=False)


def compare_all(X, y, feature_names, k=15) -> pd.DataFrame:
    return pd.DataFrame({
        "rf_importance": random_forest_importance(X, y, feature_names),
        "mutual_info": mutual_information(X, y, feature_names),
        "rfe_rank": rfe_ranking(X, y, feature_names, n_features=k),
        "select_k_best": select_k_best(X, y, feature_names, k=k),
    })
