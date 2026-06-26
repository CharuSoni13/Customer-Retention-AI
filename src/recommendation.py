"""Rule-based retention recommendation engine."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _recommend_one(prob: float, segment: str, row: pd.Series) -> dict:
    actions: list[str] = []
    priority = "Low"

    if prob >= 0.7:
        priority = "Critical"
    elif prob >= 0.5:
        priority = "High"
    elif prob >= 0.3:
        priority = "Medium"

    contract = str(row.get("Contract", ""))
    tenure = float(row.get("tenure", 0) or 0)
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    tech_sup = str(row.get("TechSupport", ""))
    payment = str(row.get("PaymentMethod", ""))

    if prob >= 0.5 and contract == "Month-to-month":
        actions.append("Offer annual contract with 15% discount")
    if prob >= 0.5 and tech_sup == "No":
        actions.append("Free TechSupport add-on for 3 months")
    if prob >= 0.5 and payment == "Electronic check":
        actions.append("Migrate to auto-pay with $5/month rebate")
    if monthly >= 80 and prob >= 0.4:
        actions.append("Dedicated customer success manager")
    if tenure < 6 and prob >= 0.3:
        actions.append("Onboarding check-in call + welcome credit")

    if segment == "Loyal Premium Customers":
        actions.append("Loyalty reward / VIP perks")
    elif segment == "High Value Customers":
        actions.append("Personalized upsell campaign")
    elif segment == "Budget Customers" and prob >= 0.4:
        actions.append("Targeted discount offer")
    elif segment == "New Customers":
        actions.append("Engagement drip campaign")
    elif segment == "High-Risk Customers":
        actions.append("Win-back offer + retention specialist outreach")

    if not actions:
        actions.append("Maintain current experience; monitor")

    revenue_at_risk = round(monthly * 12 * prob, 2)

    return {
        "priority": priority,
        "actions": actions,
        "revenue_at_risk_annual": revenue_at_risk,
    }


def recommend(
    probs: Iterable[float],
    segments: Iterable[str],
    customer_df: pd.DataFrame,
) -> pd.DataFrame:
    out = []
    probs = list(probs)
    segments = list(segments)
    for i, (_, row) in enumerate(customer_df.reset_index(drop=True).iterrows()):
        out.append(_recommend_one(probs[i], segments[i], row))
    return pd.DataFrame(out)
