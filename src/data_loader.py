"""Data loading + synthetic Telco-like dataset generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_CSV, RANDOM_STATE


def generate_synthetic_telco(n: int = 7043, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, n)
    phone = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multi_lines = np.where(
        phone == "No", "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    no_inet = internet == "No"

    def inet_service(prob_yes: float) -> np.ndarray:
        out = rng.choice(["Yes", "No"], n, p=[prob_yes, 1 - prob_yes])
        return np.where(no_inet, "No internet service", out)

    online_sec = inet_service(0.36)
    online_bk = inet_service(0.34)
    dev_prot = inet_service(0.34)
    tech_sup = inet_service(0.29)
    streaming_tv = inet_service(0.38)
    streaming_mv = inet_service(0.39)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )

    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 20))
    addon = sum((s == "Yes").astype(int) for s in [online_sec, online_bk, dev_prot, tech_sup, streaming_tv, streaming_mv])
    monthly = base + addon * 5 + rng.normal(0, 4, n)
    monthly = np.clip(monthly, 18, 120).round(2)
    total = (monthly * tenure + rng.normal(0, 30, n)).clip(0).round(2)

    # Latent churn signal: short tenure, month-to-month, fiber, electronic check, no tech support
    logit = (
        -1.4
        - 0.04 * tenure
        + 1.2 * (contract == "Month-to-month").astype(int)
        - 0.8 * (contract == "Two year").astype(int)
        + 0.5 * (internet == "Fiber optic").astype(int)
        + 0.4 * (payment == "Electronic check").astype(int)
        + 0.3 * (paperless == "Yes").astype(int)
        - 0.4 * (tech_sup == "Yes").astype(int)
        - 0.3 * (online_sec == "Yes").astype(int)
        + 0.5 * senior
        + 0.012 * (monthly - 65)
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"C{1000000 + i}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bk,
        "DeviceProtection": dev_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df


def load_data(path=None) -> pd.DataFrame:
    """Load the Telco CSV. If missing, generate a synthetic dataset and cache it."""
    path = path or RAW_CSV
    if not path.exists():
        df = generate_synthetic_telco()
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    # Telco quirk: TotalCharges may have blanks for new customers
    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df):,} rows")
    print(df["Churn"].value_counts(normalize=True))
