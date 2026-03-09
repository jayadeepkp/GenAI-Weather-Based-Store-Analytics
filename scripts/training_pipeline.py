from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for p in [ROOT, SRC]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dataset import DatasetBuilder


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + eps
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_sets(df: pd.DataFrame, target: str):
    """
    Keep engineered features from DatasetBuilder output,
    but remove current-day target-like columns to avoid leakage.
    """
    leakage_cols = ["invoice_count", "oc_count", "fleet_oc_count"]
    non_feature_cols = ["invoice_date"]

    feature_cols = [c for c in df.columns if c not in leakage_cols + non_feature_cols]

    if target not in ["invoice_count", "oc_count"]:
        raise ValueError(f"Unsupported target: {target}")

    categorical_candidates = [
        "store_id",
        "rain_bucket",
        "snow_bucket",
        "heat_bucket",
        "cold_bucket",
        "severity",
        "market_id",
        "store_state",
        "time_zone_code",
        "area_id",
        "marketing_area_id",
    ]

    categorical_cols = [c for c in categorical_candidates if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    return feature_cols, categorical_cols, numeric_cols


def main():
    parser = argparse.ArgumentParser(description="Production training pipeline for baseline HGB model")
    parser.add_argument("--target", default="invoice_count", choices=["invoice_count", "oc_count"])
    parser.add_argument("--cutoff", default="2022-01-01")
    parser.add_argument("--out", default="artifacts/training_pipeline_v1")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load DatasetBuilder output
    builder = DatasetBuilder(cutoff=args.cutoff)
    train_df = builder.train.copy()
    valid_df = builder.valid.copy()

    # Sanity checks
    if train_df.empty or valid_df.empty:
        raise ValueError("Train or validation dataframe is empty.")

    if train_df["invoice_date"].max() >= valid_df["invoice_date"].min():
        raise ValueError("Time split failed: train overlaps validation.")

    # 2) Build feature lists
    feature_cols, categorical_cols, numeric_cols = build_feature_sets(train_df, args.target)

    X_train = train_df[feature_cols].copy()
    y_train = train_df[args.target].astype(float).copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[args.target].astype(float).copy()

    # 3) Preprocess + train baseline HGB
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), categorical_cols),
            ("num", SimpleImputer(strategy="median"), numeric_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        random_state=args.random_state,
        learning_rate=0.05,
        max_iter=400,
        max_depth=None,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_valid)

    # 4) Metrics
    metrics = {
        "target": args.target,
        "cutoff": args.cutoff,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_features": int(len(feature_cols)),
        "categorical_features": categorical_cols,
        "numeric_feature_count": int(len(numeric_cols)),
        "mae": float(mean_absolute_error(y_valid, pred)),
        "mse": float(mean_squared_error(y_valid, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, pred))),
        "wape_pct": wape(y_valid.values, pred),
        "safe_mape_pct": safe_mape(y_valid.values, pred),
    }

    # 5) Save artifacts
    joblib.dump(pipe, out_dir / "model.joblib")

    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "store_id": valid_df["store_id"].values,
            "invoice_date": valid_df["invoice_date"].values,
            "actual": y_valid.values,
            "pred": pred,
        }
    )
    pred_df["abs_error"] = (pred_df["actual"] - pred_df["pred"]).abs()
    pred_df.to_csv(out_dir / "predictions_valid.csv", index=False)

    print("Saved artifacts to:", out_dir.resolve())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()