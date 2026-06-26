"""SHAP explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values_for(model, X, max_samples: int = 500):
    """Return (shap_values, base_value, sample_X) for the positive class."""
    import shap

    X_sample = X.sample(min(max_samples, len(X)), random_state=42) if hasattr(X, "sample") else X[:max_samples]

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        base = explainer.expected_value
        if isinstance(sv, list):
            sv = sv[1]
            base = base[1] if hasattr(base, "__len__") else base
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X_sample)
        except Exception:
            explainer = shap.Explainer(model.predict_proba, X_sample)
        sv_obj = explainer(X_sample)
        sv = sv_obj.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = sv_obj.base_values
        if hasattr(base, "__len__") and np.ndim(base) > 0:
            base = float(np.mean(base))

    return sv, base, X_sample


def top_drivers(shap_row: np.ndarray, feature_names: list[str], k: int = 5) -> pd.DataFrame:
    s = pd.Series(shap_row, index=feature_names)
    abs_top = s.abs().sort_values(ascending=False).head(k).index
    return s.loc[abs_top].rename("shap_value").to_frame()
