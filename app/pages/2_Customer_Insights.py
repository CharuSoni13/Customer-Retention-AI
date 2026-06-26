"""Per-customer drill-down with SHAP explanation."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import assign_segment, load_artifacts, load_processed, predict_proba  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.explainability import shap_values_for, top_drivers  # noqa: E402
from src.recommendation import recommend  # noqa: E402

st.set_page_config(page_title="Customer Insights", page_icon="🧠", layout="wide")
st.title("🧠 Customer Insights — SHAP explanation")

artifacts = load_artifacts()
df = load_processed()

if "customerID" not in df.columns:
    st.error("Processed dataset needs `customerID` to look up a customer.")
    st.stop()

cid = st.selectbox("Pick a customer", df["customerID"].head(500).tolist())
row = df[df["customerID"] == cid].head(1)

probs, eng = predict_proba(row, artifacts)
seg = assign_segment(eng, artifacts).iloc[0]
prob = float(probs[0])

c1, c2, c3 = st.columns(3)
c1.metric("Churn Probability", f"{prob:.1%}")
c2.metric("Segment", seg)
c3.metric("Monthly Charges", f"${float(row['MonthlyCharges'].iloc[0]):.2f}")

st.subheader("Top SHAP drivers")
Xt = artifacts["preprocessor"].transform(eng.drop(columns=[c for c in ("Churn", "customerID") if c in eng.columns]))
Xt_df = pd.DataFrame(Xt, columns=artifacts["feature_names"])

try:
    sv, base, _ = shap_values_for(artifacts["model"], Xt_df, max_samples=1)
    drivers = top_drivers(np.asarray(sv)[0], artifacts["feature_names"], k=8)
    drivers["impact"] = np.where(drivers["shap_value"] > 0, "↑ pushes toward churn", "↓ pushes toward retention")
    st.dataframe(drivers, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    drivers_sorted = drivers.sort_values("shap_value")
    colors = ["#16A34A" if v < 0 else "#DC2626" for v in drivers_sorted["shap_value"]]
    ax.barh(drivers_sorted.index, drivers_sorted["shap_value"], color=colors)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.axvline(0, color="black", linewidth=0.6)
    st.pyplot(fig, clear_figure=True)
except Exception as exc:  # noqa: BLE001
    st.warning(f"SHAP explanation unavailable: {exc}")

st.subheader("Retention recommendation")
rec = recommend([prob], [seg], eng).iloc[0]
st.write(f"**Priority:** {rec['priority']}  •  **Revenue at risk (annual):** ${rec['revenue_at_risk_annual']:,.2f}")
for a in rec["actions"]:
    st.markdown(f"- {a}")
