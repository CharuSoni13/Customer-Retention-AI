"""Home dashboard."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils import load_artifacts, load_processed, assign_segment, predict_proba

st.set_page_config(page_title="Customer Retention AI", page_icon="📡", layout="wide")

st.title("📡 Smart Customer Retention Intelligence")
st.caption("Predict churn, understand drivers, segment customers, recommend retention actions.")

artifacts = load_artifacts()
df = load_processed()

probs, eng = predict_proba(df, artifacts)
if "segment" not in df.columns:
    df["segment"] = assign_segment(eng, artifacts).values
df["churn_prob"] = probs

total = len(df)
churners = (df["Churn"] == "Yes").sum() if "Churn" in df.columns else int((probs >= 0.5).sum())
churn_rate = churners / total
revenue = (df["MonthlyCharges"].sum() * 12) if "MonthlyCharges" in df.columns else 0
high_risk = (df["churn_prob"] >= 0.5).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Churn Rate", f"{churn_rate:.1%}")
c3.metric("Annual Revenue", f"${revenue:,.0f}")
c4.metric("High-Risk Customers", f"{high_risk:,}")
c5.metric("Segments", df["segment"].nunique())

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Segment Distribution")
    seg_counts = df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    fig = px.pie(seg_counts, names="segment", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(df, x="churn_prob", nbins=40, color_discrete_sequence=["#EF4444"])
    fig.update_layout(xaxis_title="Predicted churn probability", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 10 At-Risk Customers")
cols = [c for c in ["customerID", "tenure", "Contract", "MonthlyCharges", "segment", "churn_prob"] if c in df.columns]
top_risk = df.sort_values("churn_prob", ascending=False).head(10)[cols]
st.dataframe(top_risk, use_container_width=True)

st.info(f"Model in production: **{artifacts['model_name']}** — use the sidebar to make predictions or browse analytics.")
