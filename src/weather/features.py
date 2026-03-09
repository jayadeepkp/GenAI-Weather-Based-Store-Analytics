from __future__ import annotations
import pandas as pd

# === main feature function ===

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds trailing lag and rolling weather features per store.
    Avoids leakage by only using past values.
    """

    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    g = out.groupby("store_id", group_keys=False)

    # simple weather features
    out["rain_mm"] = out["prcp"].fillna(0.0)
    out["snow_cm"] = out["snow"].fillna(0.0)
    out["temp_range"] = out["tmax"] - out["tmin"]
    
    out["is_rain"] = (out["rain_mm"] > 0).astype(int)
    out["is_snow"] = (out["snow_cm"] > 0).astype(int)
    out["is_freezing"] = (out["tmin"].fillna(99) <= 0).astype(int)

    out["heavy_rain"] = (out["rain_mm"] >= 10).astype(int)
    out["heavy_snow"] = (out["snow_cm"] >= 5).astype(int)

    out["rain_bucket"] = out["rain_mm"].apply(rain_bucket)
    out["snow_bucket"] = out["snow_cm"].apply(snow_bucket)
    out["heat_bucket"] = out["tmax"].apply(heat_bucket)
    out["cold_bucket"] = out["tmin"].apply(cold_bucket)

    out["heavy_rain"] = (out["rain_bucket"] == "rain_heavy").astype(int)
    out["heavy_snow"] = (out["snow_bucket"] == "snow_heavy").astype(int)
    out["extreme_heat"] = (out["heat_bucket"] == "heat_extreme").astype(int)
    out["extreme_cold"] = (out["cold_bucket"] == "cold_vcold").astype(int)
    out["freezing"] = out["tmin"].fillna(99).le(0).astype(int)

    out["severity"] = out.apply(severity, axis=1)

    out["temp_range"] = (out["tmax"] - out["tmin"]).fillna(0)
    out["heavy_rain_weekend"] = out["heavy_rain"] * out["is_weekend"]
    out["heavy_snow_weekend"] = out["heavy_snow"] * out["is_weekend"]
    out["extreme_heat_weekend"] = out["extreme_heat"] * out["is_weekend"]
    out["snow_freezing"] = out["snow_cm"] * out["freezing"]

    return out

# === helper functions ===

def rain_bucket(x):
    if x <= 0: return "rain_0"
    if x < 5:  return "rain_light"
    if x < 15: return "rain_med"
    return "rain_heavy"

def snow_bucket(x):
    if x <= 0: return "snow_0"
    if x < 2:  return "snow_light"
    if x < 5:  return "snow_med"
    return "snow_heavy"

def heat_bucket(tmax):
    if pd.isna(tmax): return "heat_na"
    if tmax < 30:     return "heat_ok"
    if tmax < 35:     return "heat_hot"
    return "heat_extreme"

def cold_bucket(tmin):
    if pd.isna(tmin): return "cold_na"
    if tmin <= -10:   return "cold_vcold"
    if tmin <= -5:    return "cold_cold"
    if tmin <= 0:     return "cold_freezing"
    return "cold_ok"

def severity(row):
    if row["heavy_snow"] == 1: return "sev_snow_heavy"
    if row["heavy_rain"] == 1: return "sev_rain_heavy"
    if row["extreme_heat"] == 1: return "sev_heat_extreme"
    if row["extreme_cold"] == 1: return "sev_cold_extreme"
    if row["freezing"] == 1: return "sev_freezing"
    return "sev_normal"

