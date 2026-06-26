"""Streamlit-side helpers: load artifacts, predict, explain."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feature_engineering, preprocessing, segmentation  # noqa: E402
from src.config import MODELS_DIR, PROCESSED_CSV  # noqa: E402


@st.cache_resource
def load_artifacts():
    path = MODELS_DIR / "artifacts.joblib"
    if not path.exists():
        st.error(
            "Model artifacts not found. Run the training pipeline first:\n\n"
            "```\npython -m src.pipeline\n```"
        )
        st.stop()
    art = joblib.load(path)
    _patch_sklearn_compat(art.get("model"))
    return art


def _patch_sklearn_compat(model) -> None:
    """Bridge sklearn version drift between training env and runtime env."""
    if model is None:
        return
    cls = type(model).__name__
    if cls == "LogisticRegression" and not hasattr(model, "multi_class"):
        # Removed in sklearn 1.7; older sklearn versions still read it in predict_proba.
        model.multi_class = "auto"
    if not hasattr(model, "n_features_in_") and hasattr(model, "coef_"):
        model.n_features_in_ = model.coef_.shape[1]


@st.cache_data
def load_processed():
    if not PROCESSED_CSV.exists():
        st.error("Processed dataset missing. Run `python -m src.pipeline`.")
        st.stop()
    return pd.read_csv(PROCESSED_CSV)


def engineer_and_transform(raw_df: pd.DataFrame, artifacts) -> tuple[pd.DataFrame, np.ndarray]:
    eng = feature_engineering.add_engineered_features(preprocessing.clean(raw_df))
    drop = [c for c in ("Churn", "customerID") if c in eng.columns]
    X = eng.drop(columns=drop)
    Xt = artifacts["preprocessor"].transform(X)
    return eng, Xt


def predict_proba(raw_df: pd.DataFrame, artifacts) -> tuple[np.ndarray, pd.DataFrame]:
    eng, Xt = engineer_and_transform(raw_df, artifacts)
    proba = artifacts["model"].predict_proba(Xt)[:, 1]
    return proba, eng


def assign_segment(eng_df: pd.DataFrame, artifacts) -> pd.Series:
    clusters = segmentation.assign_segments(
        eng_df, artifacts["kmeans"], artifacts["kmeans_scaler"], artifacts["kmeans_features"],
    )
    return clusters.map(artifacts["segment_labels"]).fillna("Unsegmented").rename("segment")
