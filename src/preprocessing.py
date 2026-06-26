"""Preprocessing: cleaning, encoding, scaling, splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COL, RANDOM_STATE, TARGET, TEST_SIZE


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df = df.drop_duplicates()
    return df


def split_xy(df: pd.DataFrame):
    target = df[TARGET]
    if pd.api.types.is_numeric_dtype(target):
        y = target.astype(int)
    else:
        y = (target.astype(str).str.strip().str.lower() == "yes").astype(int)
    drop = [c for c in (TARGET, ID_COL) if c in df.columns]
    X = df.drop(columns=drop)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            names.extend(trans.get_feature_names_out(cols).tolist())
    return names


def train_test(df: pd.DataFrame):
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
