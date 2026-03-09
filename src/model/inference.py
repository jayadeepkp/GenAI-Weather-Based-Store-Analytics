from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.dataset import DatasetBuilder


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_dir(target: str) -> Path:
    root = _project_root()

    if target == "invoice_count":
        return root / "artifacts" / "training_pipeline_invoice_v1"
    if target == "oc_count":
        return root / "artifacts" / "training_pipeline_oc_v1"

    raise ValueError("target must be 'invoice_count' or 'oc_count'")


def _load_model_and_features(target: str):
    artifact_dir = _artifact_dir(target)

    model_path = artifact_dir / "model.joblib"
    feature_cols_path = artifact_dir / "feature_cols.json"
    preds_path = artifact_dir / "predictions_valid.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature column file: {feature_cols_path}")

    model = joblib.load(model_path)

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    interval_width = None
    if preds_path.exists():
        pred_df = pd.read_csv(preds_path)
        if {"actual", "pred"}.issubset(pred_df.columns):
            residual = (pred_df["actual"] - pred_df["pred"]).abs()
            if len(residual) > 0:
                interval_width = float(residual.quantile(0.95))

    return model, feature_cols, interval_width


def _feature_alias_map(target: str) -> dict[str, str]:
    if target == "invoice_count":
        return {
            "invoice_count_lag_1": "inv_lag_1",
            "invoice_count_lag_7": "inv_lag_7",
            "invoice_count_roll7_mean": "inv_rollmean_7",
            "invoice_count_roll14_mean": "inv_rollmean_14",
            "invoice_count_roll28_mean": "inv_rollmean_28",
        }

    if target == "oc_count":
        return {
            "oc_count_lag_1": "inv_lag_1",
            "oc_count_lag_7": "inv_lag_7",
            "oc_count_lag_14": "inv_lag_14",
            "oc_count_roll7_mean": "inv_rollmean_7",
            "oc_count_roll14_mean": "inv_rollmean_14",
            "oc_count_roll28_mean": "inv_rollmean_28",

            # fallback in case saved oc model used invoice-style lag names
            "invoice_count_lag_1": "inv_lag_1",
            "invoice_count_lag_7": "inv_lag_7",
            "invoice_count_lag_14": "invoice_count_lag_14",
            "invoice_count_roll7_mean": "inv_rollmean_7",
            "invoice_count_roll14_mean": "inv_rollmean_14",
            "invoice_count_roll28_mean": "inv_rollmean_28",
        }

    return {}


def _add_legacy_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Backfill legacy feature names expected by older saved artifacts.
    We compute them from the current DatasetBuilder output when missing.
    """
    out = df.sort_values(["store_id", "invoice_date"]).copy()
    g = out.groupby("store_id", sort=False)

    # Invoice legacy lag used by older artifacts
    if "invoice_count" in out.columns and "invoice_count_lag_14" not in out.columns:
        out["invoice_count_lag_14"] = g["invoice_count"].shift(14)

    # OC legacy lag if needed
    if "oc_count" in out.columns and "oc_count_lag_14" not in out.columns:
        out["oc_count_lag_14"] = g["oc_count"].shift(14)

    return out


def _prepare_features(rows: pd.DataFrame, feature_cols: list[str], target: str) -> pd.DataFrame:
    """
    Build the exact feature matrix expected by the saved model, using alias
    mapping when current DatasetBuilder column names differ from saved artifact
    feature names.
    """
    alias_map = _feature_alias_map(target)
    X = pd.DataFrame(index=rows.index)

    missing = []
    for col in feature_cols:
        if col in rows.columns:
            X[col] = rows[col]
        elif col in alias_map and alias_map[col] in rows.columns:
            X[col] = rows[alias_map[col]]
        else:
            missing.append(col)

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return X


def predict(
    store_id: int,
    start_date: str,
    end_date: str,
    target: str = "invoice_count",
    cutoff: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Predict for an existing store/date range using DatasetBuilder output.

    Important:
    - This predicts only for rows that already exist in the built dataset.
    - It does NOT fetch future weather.
    - It does NOT create future lag features beyond the available dataset.
    """
    model, feature_cols, interval_width = _load_model_and_features(target)

    builder = DatasetBuilder(cutoff=cutoff)
    df = pd.concat([builder.train.copy(), builder.valid.copy()], ignore_index=True)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.normalize()
    df = _add_legacy_features(df, target)

    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()

    mask = (
        (df["store_id"].astype(int) == int(store_id))
        & (df["invoice_date"] >= start_dt)
        & (df["invoice_date"] <= end_dt)
    )
    rows = df.loc[mask].copy()

    if rows.empty:
        return pd.DataFrame(
            columns=["store_id", "invoice_date", "yhat", "yhat_lower", "yhat_upper"]
        )

    X = _prepare_features(rows, feature_cols, target)
    yhat = model.predict(X)

    out = rows[["store_id", "invoice_date"]].copy()
    out["yhat"] = yhat

    if interval_width is not None:
        out["yhat_lower"] = out["yhat"] - interval_width
        out["yhat_upper"] = out["yhat"] + interval_width
    else:
        out["yhat_lower"] = pd.NA
        out["yhat_upper"] = pd.NA

    return out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)