import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date

from .utils import *

# load previous model predictions for 2022
ROOT = Path(__file__).resolve().parents[1]
PREDS22 = pd.read_csv(ROOT / "data_processed/predictions_hgb_hierarchical_2022.csv")
PREDS22["invoice_date"] = pd.to_datetime(PREDS22["invoice_date"]).dt.normalize()

# constants for creating baselines and confidence intervals
(GLOBAL_LO, GLOBAL_HI), BOUNDS = build_bounds_by_severity(PREDS22)
BASELINE_OC = build_baseline(data22, "oc_count")
BASELINE_INV = build_baseline(data22, "invoice_count")

# json request/response format
class WeatherForecast(BaseModel):
    tmin: float
    tmax: float
    tavg: float
    prcp: float
    wspd: float
    snow: float

class PredictReq(BaseModel):
    store_id: int
    date: date
    weather: WeatherForecast | None = None

class PredictRes(BaseModel):
    store_id: int
    date: date
    pred_invoice: float
    pred_oc: float
    oc_lo95: float
    oc_hi95: float
    severity: str
    oc_pchg: float
    actual_invoice: float | None = None
    actual_oc: float | None = None

# === main api logic ===
    
app = FastAPI()

@app.post("/predict")
async def get_prediction(req: PredictReq):
    # check store_id
    sid = req.store_id
    if sid not in PREDS22["store_id"].unique():
        raise HTTPException(status_code=400, detail="Predictions not available for that store ID.")

    # check date
    date = None
    try:
        date = pd.to_datetime(req.date).normalize()
    except:
        raise HTTPException(status_code=400, detail="'date' cannot be interpreted as a date.")

    if date < PREDS22.invoice_date.min():
        raise HTTPException(status_code=400,
                            detail="Prediction not available. Please select a date on or after 2022-01-02.")
    
    res = {"store_id": sid, "date": req.date}
    
    row = PREDS22[(PREDS22["store_id"] == sid) & (PREDS22["invoice_date"] == date)]

    if not row.empty:
        # cache hit
        r = row.iloc[0]
        res["pred_invoice"] = float(r["pred_invoice"])
        res["pred_oc"] = float(r["pred_oc_total"])
        res["oc_lo95"] = float(r["oc_lower_95"])
        res["oc_hi95"] = float(r["oc_upper_95"])
        res["severity"] = str(r["severity"])

        baseline_oc = get_baseline(BASELINE_OC, sid, date, fallback=float(PREDS22["oc_count"].mean()))
        res["oc_pchg"] = pct_change(res["pred_oc"], baseline_oc)

        res["actual_invoice"] = float(r["invoice_count"])
        res["actual_oc"] = float(r["oc_count"])
    else:
        # cache miss, use models to forecast
        if req.weather == None:
            raise HTTPException(status_code=400, detail="Prediction not available, nor was weather data provided in request body.")

        X = build_forecast_features(req)
        preds = predict_all(X)

        res["pred_invoice"] = preds["invoice_pred"]
        res["pred_oc"] = preds["oc_total_pred"]

        res["severity"] = X.iloc[0]["severity"]

        baseline_oc = get_baseline(BASELINE_OC, sid, date, fallback=float(PREDS22["oc_count"].mean()))
        res["oc_pchg"] = pct_change(res["pred_oc"], baseline_oc)

        # 95% OC interval using severity-specific residual bounds from 2022
        qlo, qhi = BOUNDS.get(res["severity"], (GLOBAL_LO, GLOBAL_HI))
        res["oc_lo95"] = max(res["pred_oc"] + qlo, 0.0)
        res["oc_hi95"] = max(res["pred_oc"] + qhi, 0.0)


    return PredictRes(**res)
