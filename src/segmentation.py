"""Customer segmentation via KMeans + business label mapping."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "ServiceCount", "ContractRiskScore"]


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 5, features=None) -> tuple[KMeans, StandardScaler, list[str]]:
    features = features or [c for c in SEGMENT_FEATURES if c in df.columns]
    X = df[features].fillna(0).values
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km.fit(scaler.transform(X))
    return km, scaler, features


def assign_segments(df: pd.DataFrame, km: KMeans, scaler: StandardScaler, features: list[str]) -> pd.Series:
    X = df[features].fillna(0).values
    return pd.Series(km.predict(scaler.transform(X)), index=df.index, name="cluster")


def label_segments(df: pd.DataFrame, clusters: pd.Series) -> pd.Series:
    """Map cluster id → business-friendly label using cluster centroids on raw features."""
    work = df.copy()
    work["cluster"] = clusters.values
    summary = work.groupby("cluster").agg(
        tenure=("tenure", "mean"),
        monthly=("MonthlyCharges", "mean"),
        total=("TotalCharges", "mean"),
        risk=("ContractRiskScore", "mean") if "ContractRiskScore" in work.columns else ("tenure", "mean"),
    )

    labels = {}
    for cid, row in summary.iterrows():
        if row["tenure"] < 12:
            labels[cid] = "New Customers"
        elif row["monthly"] > summary["monthly"].quantile(0.75) and row["tenure"] > summary["tenure"].median():
            labels[cid] = "Loyal Premium Customers"
        elif row["monthly"] > summary["monthly"].quantile(0.75):
            labels[cid] = "High Value Customers"
        elif "risk" in row and row["risk"] >= summary["risk"].quantile(0.75):
            labels[cid] = "High-Risk Customers"
        else:
            labels[cid] = "Budget Customers"

    return clusters.map(labels).rename("segment")
