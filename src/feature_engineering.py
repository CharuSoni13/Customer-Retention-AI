"""Feature engineering for Telco churn data."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CONTRACT_RISK = {"Month-to-month": 3, "One year": 2, "Two year": 1}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    tenure_safe = df["tenure"].replace(0, 1)
    df["AvgMonthlySpend"] = (df["TotalCharges"].fillna(0) / tenure_safe).round(2)
    df["CustomerLifetimeValue"] = (df["MonthlyCharges"] * df["tenure"]).round(2)

    df["ServiceCount"] = sum(
        df[c].isin(["Yes"]).astype(int) for c in SERVICE_COLS if c in df.columns
    )

    df["IsFamilyCustomer"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    digital_pay = ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"]
    df["IsDigitalCustomer"] = (
        (df["PaperlessBilling"] == "Yes") & (df["PaymentMethod"].isin(digital_pay))
    ).astype(int)

    df["ContractRiskScore"] = df["Contract"].map(CONTRACT_RISK).fillna(2).astype(int)

    df["TenureBucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 24, 48, 100],
        labels=["0-6m", "7-24m", "25-48m", "49m+"],
    ).astype(str)

    return df
