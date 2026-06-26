"""Business analytics: revenue at risk, segments, CLV, tenure trends."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import assign_segment, load_artifacts, load_processed, predict_proba  # noqa: E402

st.set_page_config(page_title="Business Analytics", page_icon="📊", layout="wide")
st.title("📊 Business Analytics")

artifacts = load_artifacts()
df = load_processed()

probs, eng = predict_proba(df, artifacts)
if "segment" not in df.columns:
    df["segment"] = assign_segment(eng, artifacts).values
df["churn_prob"] = probs
df["AnnualRevenue"] = df["MonthlyCharges"] * 12
df["RevenueAtRisk"] = df["AnnualRevenue"] * df["churn_prob"]

c1, c2, c3 = st.columns(3)
c1.metric("Annual Revenue", f"${df['AnnualRevenue'].sum():,.0f}")
c2.metric("Revenue at Risk", f"${df['RevenueAtRisk'].sum():,.0f}")
c3.metric("% At Risk", f"{df['RevenueAtRisk'].sum() / max(df['AnnualRevenue'].sum(), 1):.1%}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Revenue at risk by segment")
    seg_rev = df.groupby("segment")["RevenueAtRisk"].sum().reset_index().sort_values("RevenueAtRisk", ascending=False)
    fig = px.bar(seg_rev, x="segment", y="RevenueAtRisk", color="segment")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Customer Lifetime Value distribution")
    if "CustomerLifetimeValue" in df.columns:
        fig = px.box(df, x="segment", y="CustomerLifetimeValue", color="segment")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Churn rate by tenure bucket")
if "TenureBucket" in df.columns:
    tb = df.groupby("TenureBucket")["churn_prob"].mean().reset_index()
    fig = px.bar(tb, x="TenureBucket", y="churn_prob", labels={"churn_prob": "Avg predicted churn"})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly charges vs tenure (size = churn probability)")
fig = px.scatter(
    df.sample(min(2000, len(df)), random_state=1),
    x="tenure", y="MonthlyCharges", color="segment", size="churn_prob",
    hover_data=["churn_prob"], opacity=0.7,
)
st.plotly_chart(fig, use_container_width=True)
