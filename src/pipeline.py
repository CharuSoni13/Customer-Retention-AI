"""End-to-end training pipeline: load → engineer → preprocess → train → eval → segment → save."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import data_loader, evaluation, feature_engineering, model_training, preprocessing, segmentation
from .config import MODELS_DIR, PROCESSED_CSV, REPORTS_DIR


def run() -> dict:
    print("[1/7] Loading data...")
    df = data_loader.load_data()

    print("[2/7] Cleaning + feature engineering...")
    df = preprocessing.clean(df)
    df = feature_engineering.add_engineered_features(df)
    df.to_csv(PROCESSED_CSV, index=False)

    print("[3/7] Train/test split + preprocessing...")
    X_train, X_test, y_train, y_test = preprocessing.train_test(df)
    pre = preprocessing.build_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = preprocessing.get_feature_names(pre)

    print("[4/7] Training models...")
    models = model_training.build_models()
    fitted = model_training.fit_all(models, X_train_t, y_train)

    print("[5/7] Evaluating...")
    leaderboard = evaluation.compare_models(fitted, X_test_t, y_test)
    print(leaderboard.to_string(index=False))
    best_name = leaderboard.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"-> Best: {best_name}")

    print("[6/7] Customer segmentation...")
    km, scaler, seg_features = segmentation.fit_kmeans(df, n_clusters=5)
    clusters = segmentation.assign_segments(df, km, scaler, seg_features)
    labels = segmentation.label_segments(df, clusters)
    df_with_seg = df.assign(cluster=clusters.values, segment=labels.values)
    df_with_seg.to_csv(PROCESSED_CSV, index=False)

    print("[7/7] Saving artifacts...")
    artifacts = {
        "preprocessor": pre,
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "kmeans": km,
        "kmeans_scaler": scaler,
        "kmeans_features": seg_features,
        "segment_labels": dict(zip(clusters, labels)),
    }
    joblib.dump(artifacts, MODELS_DIR / "artifacts.joblib")
    leaderboard.to_csv(REPORTS_DIR / "leaderboard.csv", index=False)
    with open(REPORTS_DIR / "summary.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics": leaderboard.iloc[0].drop("model").to_dict(),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(len(feature_names)),
        }, f, indent=2)

    print(f"Saved artifacts → {MODELS_DIR / 'artifacts.joblib'}")
    return {"best": best_name, "leaderboard": leaderboard}


if __name__ == "__main__":
    run()
