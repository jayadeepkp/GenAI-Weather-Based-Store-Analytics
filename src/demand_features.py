import pandas as pd

default_demand_features = ["inv_lag_1", "inv_lag_7", "inv_rollmean_7", "inv_rollmean_14", "inv_rollmean_28"]

def add_demand_features(df: pd.DataFrame, features=default_demand_features):
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    if features == []:
        return out
    
    g = out.groupby("store_id", group_keys=False)["invoice_count"]

    lag_days = [int(f.removeprefix("inv_lag_")) for f in features if f.startswith("inv_lag_")]
    rollmean_days = [int(f.removeprefix("inv_rollmean_")) for f in features if f.startswith("inv_rollmean_")]

    for i in lag_days:
        out[f"inv_lag_{i}"] = g.shift(i)

    for i in rollmean_days:
        out[f"inv_rollmean_{i}"] = g.shift(1).rolling(i).mean()
        
    return out
