import pandas as pd
import numpy as np

def add_store_features(df: pd.DataFrame):
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    day_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    out["closed_day_description"] = out["closed_day_description"].fillna("").astype(str).str.strip().str.lower()
    out["closed_dow"] = out["closed_day_description"].map(day_map)
    out["is_closed_day"] = ((out["closed_dow"].notna()) & (out["dow"] == out["closed_dow"])).astype(int)
    
    out["bay_count"] = pd.to_numeric(out["bay_count"], errors="coerce")
    out["bay_count"] = out["bay_count"].fillna(out["bay_count"].median())
    out["bay_count_log"] = np.log1p(out["bay_count"])
    
    out["is_peak_day"] = out["dow"].isin([3,4,5]).astype(int)
    out["capacity_pressure"] = out["is_peak_day"] * (1.0 / out["bay_count"])

    return out
