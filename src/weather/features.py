from __future__ import annotations
import pandas as pd


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds trailing lag and rolling weather features per store.
    Avoids leakage by only using past values.
    """

    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    g = out.groupby("store_id", group_keys=False)

    def lag(col: str, k: int):
        if col in out.columns:
            out[f"{col}_lag{k}"] = g[col].shift(k)

    def roll_mean(col: str, w: int):
        if col in out.columns:
            out[f"{col}_rollmean_{w}"] = g[col].shift(1).rolling(w).mean()

    def roll_sum(col: str, w: int):
        if col in out.columns:
            out[f"{col}_rollsum_{w}"] = g[col].shift(1).rolling(w).sum()

    def anomaly(col: str, w: int = 30):
        if col in out.columns:
            baseline = g[col].shift(1).rolling(w).mean()
            out[f"{col}_anom_{w}"] = out[col] - baseline

    for c in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd"]:
        lag(c, 1)
        lag(c, 3)
        lag(c, 7)

    for w in [3, 7]:
        roll_mean("tavg", w)
        roll_sum("prcp", w)

    anomaly("tavg", 30)

    # calendar features
    out["dow"] = out["invoice_date"].dt.dayofweek
    out["month"] = out["invoice_date"].dt.month
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)

    return out