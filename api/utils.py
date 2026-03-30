# utility functions to help with main api functionality
# (mostly borrowed from demo/app.py)

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

MODEL_DIR = ROOT / "models"
MODELS = {
    "invoice": joblib.load(MODEL_DIR / "model_hgb_invoice.joblib"),
    "non_fleet_oc": joblib.load(MODEL_DIR / "model_hgb_non_fleet_oc.joblib"),
    "fleet_oc": joblib.load(MODEL_DIR / "model_hgb_fleet_oc.joblib")
}

DATA_PROCESSED = ROOT / "data_processed"

# pull the 2022 validation dataset
data22 = None

if Path(DATA_PROCESSED / "valid_model_table.parquet").exists():
    data22 = pd.read_parquet(DATA_PROCESSED / "valid_model_table.parquet")
elif Path(DATA_PROCESSED / "model_table.parquet").exists():
    data22 = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
else:
    raise FileNotFoundError("API module requires \
    data_processed/valid_model_table.parquet or \
    data_processed/model_table.parquet to function. Please run \
    scripts/build_dataset.py first and try again.")

data22["invoice_date"] = pd.to_datetime(data22["invoice_date"]).dt.normalize()
data22 = data22[data22.invoice_date.dt.year == 2022]

# to help with build_forecast_features()
store_info = pd.read_csv(ROOT / "data_raw/store_info.csv")
store_info = store_info.sort_values(["store_id"]).reset_index(drop=True)

from src.store_features import add_store_features
from src.calendar_features import add_calendar_features
from src.weather.features import add_weather_features

# === utilities for generating predictions ===

# build a DataFrame for model predictions
def build_forecast_features(req):
    body = req.model_dump()
    body |= body["weather"]
    del body["weather"]
    df = pd.DataFrame([body])

    df = df.rename(columns={"date": "invoice_date"})
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.normalize()
    df = df.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    df = df.merge(store_info, on="store_id", how="left")
    
    df = add_calendar_features(df)
    df = add_store_features(df)
    df = add_weather_features(df)
    df["heavy_rain_capacity"] = df["heavy_rain"] * df["capacity_pressure"]
    df["heavy_snow_capacity"] = df["heavy_snow"] * df["capacity_pressure"]
    df["freezing_capacity"] = df["freezing"] * df["capacity_pressure"]
    df["extreme_heat_capacity"] = df["extreme_heat"] * df["capacity_pressure"]
    df["extreme_cold_capacity"] = df["extreme_cold"] * df["capacity_pressure"]

    # bulid lag/roll columns based 2022 data
    row = df.iloc[0]
    hist = data22[data22.store_id == row.store_id].sort_values("invoice_date")
    last = hist.iloc[-1] if not hist.empty else None

    same_date_22 = pd.Timestamp(2022, row.invoice_date.month, row.invoice_date.day)
    row_22 = hist[hist.invoice_date == same_date_22] if not hist.empty else None
    row_22 = row_22.iloc[0] if not row_22.empty else None

    store_mean_inv = float(hist["invoice_count"].mean()) if ("invoice_count" in hist.columns and not hist.empty) else 0.0
    
    for c in data22.columns:
        if c in row:
            continue

        if ("tmin" in c) or ("tmax" in c):
            # fill missing weather data based on monthly and daily weather data from 2022 
            if row_22 is None or c not in hist.columns or pd.isna(row_22[c]):
                store_monthly_avg = hist[hist.invoice_date.month == row.invoice_date.month][c].mean()
                row[c] = float(store_monthly_avg)
            else:
                row[c] = float(row_22[c])
        elif ("lag_" in c) or ("roll" in c):
            # otherwise, use overall data for other columns
            if last is None or c not in hist.columns or pd.isna(last[c]):
                row[c] = store_mean_inv
            else:
                row[c] = float(last[c])
        else:
            row[c] = 0.0
                

    # return if column names don't need to be renamed for current models
    model_features = MODELS["invoice"].feature_names_in_
    if set(model_features) == set(row.index):
        row = row[[f for f in model_features]]
        return pd.DataFrame([row])

    # otherwise, rename columns
    alias_map = {}
    temp_p = re.compile(r"(tmin|tmax)(.*)")
    rollmean_p = re.compile(r"(.+)rollmean_(\d+)")
    for c in row.index:
        # first check temperature columns
        m = temp_p.match(c)
        if m:
            alias_map[c] = temp_p.sub(r"\1_c\2", c)

        # then check rolling mean columns
        m = rollmean_p.match(c)
        if c in alias_map and m:
            alias_map[c] = rollmean_p.sub(r"\1roll\2_mean", alias_map[c])
        elif m:
            alias_map[c] = rollmean_p.sub(r"\1roll\2_mean", c)

    row = row.rename(alias_map)
    row = row[[f for f in model_features]]
    
    return pd.DataFrame([row])

# make predictions using all current models
def predict_all(X: pd.DataFrame) -> dict:
    inv = float(np.clip(MODELS["invoice"].predict(X)[0], 0, None))
    nf  = float(np.clip(MODELS["non_fleet_oc"].predict(X)[0], 0, None))
    fl  = float(np.clip(MODELS["fleet_oc"].predict(X)[0], 0, None))
    oc = nf + fl

    oc = min(oc, inv)
    fl = min(fl, oc)
    nf = max(oc - fl, 0.0)

    return {"invoice_pred": inv, "non_fleet_oc_pred": nf, "fleet_oc_pred": fl, "oc_total_pred": oc}


# === utilities for generating baselines, confidence intervals, etc. ===

def build_baseline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    base = (
        df.groupby(["store_id", "dow", "month"])[target_col]
        .mean()
        .rename("baseline")
        .reset_index()
    )
    return base


def get_baseline(base_df: pd.DataFrame, store_id: int, date: pd.Timestamp, fallback: float) -> float:
    dow = int(date.dayofweek)
    month = int(date.month)
    b = base_df[(base_df["store_id"] == store_id) & (base_df["dow"] == dow) & (base_df["month"] == month)]
    if b.empty:
        return float(fallback)
    return float(b["baseline"].iloc[0])

# percent change from baseline
def pct_change(pred: float, baseline: float) -> float:
    if baseline is None or not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return (pred - baseline) / baseline * 100.0

# severity + interval residual bounds
def build_bounds_by_severity(pred22: pd.DataFrame):
    resid = (pred22["oc_count"].astype(float) - pred22["pred_oc_total"].astype(float)).replace([np.inf,-np.inf], np.nan).dropna()
    global_lo = float(np.percentile(resid, 2.5))
    global_hi = float(np.percentile(resid, 97.5))

    bounds = {}
    for sev, grp in pred22.groupby("severity"):
        r = (grp["oc_count"].astype(float) - grp["pred_oc_total"].astype(float)).replace([np.inf,-np.inf], np.nan).dropna()
        if len(r) >= 200:
            bounds[sev] = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
        else:
            bounds[sev] = (global_lo, global_hi)

    return (global_lo, global_hi), bounds
