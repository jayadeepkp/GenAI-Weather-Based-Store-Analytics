import pandas as pd

def add_demand_features(df: pd.DataFrame):
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    out["non_fleet_oc"] = out.oc_count - out.fleet_oc_count
    return out
