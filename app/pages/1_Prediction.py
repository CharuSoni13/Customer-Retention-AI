"""Single + batch churn prediction."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import assign_segment, load_artifacts, predict_proba  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.recommendation import recommend  # noqa: E402

st.set_page_config(page_title="Predict Churn", page_icon="🔮", layout="wide")
st.title("🔮 Churn Prediction")

artifacts = load_artifacts()

tab1, tab2 = st.tabs(["Single Customer", "Batch (CSV)"])

with tab1:
    st.subheader("Enter customer details")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        dev_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with c3:
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    c4, c5 = st.columns(2)
    monthly = c4.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = c5.number_input("Total Charges", 0.0, 20000.0, float(monthly * max(tenure, 1)))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "customerID": "ADHOC", "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
            "DeviceProtection": dev_prot, "TechSupport": tech_sup,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        probs, eng = predict_proba(row, artifacts)
        segs = assign_segment(eng, artifacts)
        prob = float(probs[0])
        seg = segs.iloc[0]

        cA, cB, cC = st.columns(3)
        cA.metric("Churn Probability", f"{prob:.1%}")
        cB.metric("Segment", seg)
        cC.metric("Risk Level", "🔴 High" if prob >= 0.5 else ("🟡 Medium" if prob >= 0.3 else "🟢 Low"))

        rec = recommend([prob], [seg], eng).iloc[0]
        st.subheader("Recommended Actions")
        st.write(f"**Priority:** {rec['priority']}  •  **Revenue at risk (annual):** ${rec['revenue_at_risk_annual']:,.2f}")
        for a in rec["actions"]:
            st.markdown(f"- {a}")

with tab2:
    st.subheader("Upload a CSV with Telco customer columns")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        probs, eng = predict_proba(raw, artifacts)
        segs = assign_segment(eng, artifacts)
        recs = recommend(probs, segs, eng)
        out = eng.copy()
        out["churn_prob"] = probs
        out["segment"] = segs.values
        out["priority"] = recs["priority"].values
        out["recommended_actions"] = recs["actions"].apply(lambda xs: "; ".join(xs)).values
        out["revenue_at_risk_annual"] = recs["revenue_at_risk_annual"].values
        show_cols = [c for c in ["customerID", "tenure", "Contract", "MonthlyCharges", "segment", "churn_prob", "priority", "recommended_actions"] if c in out.columns]
        st.dataframe(out[show_cols].sort_values("churn_prob", ascending=False), use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode(),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
