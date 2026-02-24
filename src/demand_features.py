import pandas as pd

def add_demand_features(df: pd.DataFrame):
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    g = out.groupby("store_id", group_keys=False)

    def lag(col: str, k: int):
        if col in out.columns:
            out[f"{col}_lag_{k}"] = g[col].shift(k)

    def roll_mean(col: str, w: int):
        if col in out.columns:
            out[f"{col}_roll{w}_mean"] = g[col].shift(1).rolling(w).mean()

    lag("invoice_count", 1)
    lag("invoice_count", 7)
    lag("invoice_count", 14)

    roll_mean("invoice_count", 7)
    roll_mean("invoice_count", 14)
    roll_mean("invoice_count", 28)

    return out
