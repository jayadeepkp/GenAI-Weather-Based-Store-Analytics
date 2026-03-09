import pandas as pd


def add_lagroll(df, lagroll_features):
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    g = out.groupby("store_id", group_keys=False)


    def lag(col: str, k: int):
        if col in out.columns:
            out[f"{col}_lag_{k}"] = g[col].shift(k)
            
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
    
    for feature in lagroll_features:
        feature_split = feature.split("_")
        days = int(feature_split[-1])
        feature_type = feature_split[-2]
        base_feature = "_".join(feature_split[0:-2])

        match feature_type:
            case "lag": lag(base_feature, days)
            case "rollmean": roll_mean(base_feature, days)
            case "rollsum": roll_sum(base_feature, days)
            case "anom": anomaly(base_feature, days)

    return out
