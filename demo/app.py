# demo/app.py  (FULL FILE) — Sprint 2 (B) Hierarchical + Overall Visits + Forecast Mode + Ollama Chat
#
# WHAT THIS APP DOES
# 1) 2022 Lookup (Chat): uses predictions_hgb_hierarchical_2022.csv for OC + 95% interval + severity
#    and also shows actual invoice_count from demo_features_2022.parquet.
#
# 2) Forecast Mode (Future): uses 3 saved models:
#    - models/model_hgb_invoice.joblib
#    - models/model_hgb_non_fleet_oc.joblib
#    - models/model_hgb_fleet_oc.joblib
#    It builds a feature row (matching feature_cols_hgb_allstores.json) from forecast weather inputs and last-known lags,
#    then predicts invoice + OC split and enforces:
#       oc_total <= invoice_total
#       fleet_oc <= oc_total
#
# 3) Normal chat: everything else goes to Ollama.
#
# REQUIRED FILES
# - models/model_hgb_invoice.joblib
# - models/model_hgb_non_fleet_oc.joblib
# - models/model_hgb_fleet_oc.joblib
# - data_processed/feature_cols_hgb_allstores.json
# - data_processed/demo_features_2022.parquet
# - data_processed/predictions_hgb_hierarchical_2022.csv

import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import streamlit as st

# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
MODELS_DIR = ROOT / "models"

MODEL_INVOICE = MODELS_DIR / "model_hgb_invoice.joblib"
MODEL_NONFLEET = MODELS_DIR / "model_hgb_non_fleet_oc.joblib"
MODEL_FLEET = MODELS_DIR / "model_hgb_fleet_oc.joblib"

FEATURES_PATH = DATA_PROCESSED / "feature_cols_hgb_allstores.json"
DEMO_TABLE = DATA_PROCESSED / "demo_features_2022.parquet"
PRED_2022 = DATA_PROCESSED / "predictions_hgb_hierarchical_2022.csv"

CHAT_SYSTEM = """
You are a helpful store operations assistant.
Speak in simple, clear language.
If the user asks for a prediction, provide a short explanation.
Keep answers practical and not too long.
"""

EXTRACT_SYSTEM = """
Return ONLY valid JSON in this exact format:
{"store_id": 79609, "invoice_date": "2022-01-06"}

Rules:
- If store_id is missing, return {"error":"missing_store_id"}
- If invoice_date is missing, return {"error":"missing_invoice_date"}
- Never include any extra text outside JSON.
"""

# ----------------------------
# Ollama helpers
# ----------------------------
def ollama_chat(user_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM.strip()},
            {"role": "user", "content": user_text.strip()},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def ollama_extract(user_text: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM.strip()},
            {"role": "user", "content": user_text.strip()},
        ],
        "stream": False,
        "options": {"temperature": 0},
        "format": "json",
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start:end+1])
    return {"error": "bad_json", "raw": content}

def looks_like_prediction_request(text: str) -> bool:
    t = text.lower()
    triggers = ["predict", "prediction", "forecast", "estimate", "expected", "store", "2022-"]
    return any(x in t for x in triggers)

# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    for p in [MODEL_INVOICE, MODEL_NONFLEET, MODEL_FLEET, FEATURES_PATH, DEMO_TABLE, PRED_2022]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    model_invoice = joblib.load(MODEL_INVOICE)
    model_nonfleet = joblib.load(MODEL_NONFLEET)
    model_fleet = joblib.load(MODEL_FLEET)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    demo = pd.read_parquet(DEMO_TABLE)
    demo["invoice_date"] = pd.to_datetime(demo["invoice_date"]).dt.normalize()
    demo["store_id"] = demo["store_id"].astype(int)

    # Ensure calendar cols exist
    demo["dow"] = demo["invoice_date"].dt.dayofweek
    demo["month"] = demo["invoice_date"].dt.month
    demo["day_of_year"] = demo["invoice_date"].dt.dayofyear
    demo["year"] = demo["invoice_date"].dt.year
    demo["is_weekend"] = (demo["dow"] >= 5).astype(int)

    pred22 = pd.read_csv(PRED_2022)
    pred22["invoice_date"] = pd.to_datetime(pred22["invoice_date"]).dt.normalize()
    pred22["store_id"] = pred22["store_id"].astype(int)

    return model_invoice, model_nonfleet, model_fleet, feature_cols, demo, pred22

model_invoice, model_nonfleet, model_fleet, feature_cols, demo, pred22 = load_artifacts()

ALL_STORES = sorted(pred22["store_id"].dropna().astype(int).unique().tolist())

# ----------------------------
# Baseline for "% change vs normal"
# store + dow + month
# ----------------------------
@st.cache_data
def build_baseline(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    base = (
        df.groupby(["store_id", "dow", "month"])[target_col]
        .mean()
        .rename("baseline")
        .reset_index()
    )
    return base

BASELINE_OC = build_baseline(demo, "oc_count")
BASELINE_INV = build_baseline(demo, "invoice_count")

def get_baseline(base_df: pd.DataFrame, store_id: int, date: pd.Timestamp, fallback: float) -> float:
    dow = int(date.dayofweek)
    month = int(date.month)
    b = base_df[(base_df["store_id"] == store_id) & (base_df["dow"] == dow) & (base_df["month"] == month)]
    if b.empty:
        return float(fallback)
    return float(b["baseline"].iloc[0])

def pct_change(pred: float, baseline: float) -> float:
    if baseline is None or not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return (pred - baseline) / baseline * 100.0

# ----------------------------
# Severity + interval residual bounds (from 2022 hierarchical)
# ----------------------------
@st.cache_data
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

(GLOBAL_LO, GLOBAL_HI), BOUNDS = build_bounds_by_severity(pred22)

def severity_bucket(tmin, tmax, prcp, snow):
    heavy_rain = prcp >= 15
    heavy_snow = snow >= 5
    extreme_heat = tmax >= 35
    extreme_cold = tmin <= -10
    freezing = tmin <= 0

    if heavy_snow: return "sev_snow_heavy"
    if heavy_rain: return "sev_rain_heavy"
    if extreme_heat: return "sev_heat_extreme"
    if extreme_cold: return "sev_cold_extreme"
    if freezing: return "sev_freezing"
    return "sev_normal"

def format_sev(sev: str) -> str:
    mapping = {
        "sev_normal": "normal conditions",
        "sev_freezing": "freezing conditions",
        "sev_rain_heavy": "heavy rain",
        "sev_snow_heavy": "heavy snow",
        "sev_heat_extreme": "extreme heat",
        "sev_cold_extreme": "extreme cold",
    }
    return mapping.get(sev, sev)

# ----------------------------
# Build forecast feature row (matches your feature cols)
# Fill lag/roll from last-known store history in demo table
# ----------------------------
def build_forecast_features(store_id: int, date: pd.Timestamp, tmin, tmax, tavg, prcp, wspd, snow) -> pd.DataFrame:
    d = pd.to_datetime(date).normalize()
    dow = int(d.dayofweek)
    month = int(d.month)
    day_of_year = int(d.dayofyear)
    year = int(d.year)
    is_weekend = int(dow >= 5)

    hist = demo[demo["store_id"] == store_id].sort_values("invoice_date")
    last = hist.iloc[-1] if not hist.empty else None
    store_mean_inv = float(hist["invoice_count"].mean()) if ("invoice_count" in hist.columns and not hist.empty) else 0.0

    temp_range = float(tmax - tmin)
    rain_mm = float(prcp)
    snow_cm = float(snow)
    is_rain = int(prcp > 0)
    is_snow = int(snow > 0)
    is_freezing = int(tmin <= 0)
    heavy_rain = int(prcp >= 15)
    heavy_snow = int(snow >= 5)
    severe_weather = int(heavy_rain or heavy_snow or is_freezing)

    row = {}
    def set_if(col, val):
        if col in feature_cols:
            row[col] = val

    set_if("store_id", int(store_id))
    set_if("dow", dow)
    set_if("month", month)
    set_if("day_of_year", day_of_year)
    set_if("year", year)
    set_if("is_weekend", is_weekend)
    set_if("time_index", int(len(hist)))

    set_if("tmin", float(tmin))
    set_if("tmax", float(tmax))
    set_if("tavg", float(tavg))
    set_if("prcp", float(prcp))
    set_if("wspd", float(wspd))
    set_if("snow", float(snow))

    set_if("temp_range", temp_range)
    set_if("rain_mm", rain_mm)
    set_if("snow_cm", snow_cm)

    set_if("is_rain", is_rain)
    set_if("is_snow", is_snow)
    set_if("is_freezing", is_freezing)
    set_if("heavy_rain", heavy_rain)
    set_if("heavy_snow", heavy_snow)
    set_if("severe_weather", severe_weather)

    for c in feature_cols:
        if c in row:
            continue
        if ("lag_" in c) or ("roll" in c):
            if last is not None and c in hist.columns and pd.notna(last[c]):
                row[c] = float(last[c])
            else:
                row[c] = store_mean_inv
        else:
            row[c] = 0.0

    return pd.DataFrame([row], columns=feature_cols)

# ----------------------------
# Predict invoice + OC split and enforce hierarchy constraints
# ----------------------------
def predict_all(X: pd.DataFrame) -> dict:
    inv = float(np.clip(model_invoice.predict(X)[0], 0, None))
    nf  = float(np.clip(model_nonfleet.predict(X)[0], 0, None))
    fl  = float(np.clip(model_fleet.predict(X)[0], 0, None))
    oc = nf + fl

    oc = min(oc, inv)
    fl = min(fl, oc)
    nf = max(oc - fl, 0.0)

    return {"invoice_pred": inv, "non_fleet_oc_pred": nf, "fleet_oc_pred": fl, "oc_total_pred": oc}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Store Manager Assistant", page_icon="🛠️", layout="centered")
st.title("🛠️ Store Manager Assistant (Sprint 2: Overall + Hierarchical OC)")
st.caption("2022 lookup: **Predict store 79609 on 2022-01-04**. Future predictions: use Forecast tab.")

tab_chat, tab_forecast = st.tabs(["Chat (2022 lookup)", "Forecast Mode (Future)"])

# Tab 1: 2022 lookup
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask for a 2022 prediction like: **Predict store 79609 on 2022-01-04**."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Type your message...")

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        if not looks_like_prediction_request(user_text):
            try:
                reply = ollama_chat(user_text)
            except Exception as e:
                reply = f"Error talking to Ollama: {e}\n\nTip: run `ollama serve`."
        else:
            req = ollama_extract(user_text)

            if req.get("error") == "missing_store_id":
                reply = "Please provide a store_id (example: **Predict store 79609 on 2022-01-04**)."
            elif req.get("error") == "missing_invoice_date":
                reply = "Please provide a 2022 date like **YYYY-MM-DD**."
            elif "error" in req:
                reply = "I couldn't extract store/date. Try: **Predict store 79609 on 2022-01-04**."
            else:
                try:
                    sid = int(req["store_id"])
                    d = pd.to_datetime(req["invoice_date"]).normalize()

                    if d.year != 2022:
                        reply = "For future dates, use the **Forecast Mode** tab. This chat lookup is for 2022 validation dates."
                    else:
                        row = pred22[(pred22["store_id"] == sid) & (pred22["invoice_date"] == d)]
                        if row.empty:
                            available = pred22[pred22["store_id"] == sid]["invoice_date"].drop_duplicates().sort_values()
                            hint = available.head(5).dt.strftime("%Y-%m-%d").tolist()
                            reply = f"No 2022 row for store **{sid}** on **{d.date()}**. Try: {hint}"
                        else:
                            r = row.iloc[0]
                            pred_oc = float(r["pred_oc_total"])
                            lo95 = float(r["oc_lower_95"])
                            hi95 = float(r["oc_upper_95"])
                            sev = str(r["severity"])
                            actual_oc = float(r["oc_count"])

                            # also show actual invoice_count from demo table
                            demo_row = demo[(demo["store_id"] == sid) & (demo["invoice_date"] == d)]
                            actual_inv = float(demo_row["invoice_count"].iloc[0]) if not demo_row.empty else None

                            baseline_oc = get_baseline(BASELINE_OC, sid, d, fallback=float(demo["oc_count"].mean()))
                            pchg = pct_change(pred_oc, baseline_oc)
                            direction = "fewer" if pchg < 0 else "more"

                            reply = (
                                f"Based on {format_sev(sev)} on **{d.date()}**, store **{sid}** should see about "
                                f"**{abs(pchg):.1f}% {direction} oil-change visits** than a typical {d.strftime('%A')} in {d.strftime('%B')}.\n\n"
                                f"**Prediction (OC total):** **{pred_oc:.1f}**\n\n"
                                f"**Typical 95% range:** {lo95:.1f}–{hi95:.1f}\n\n"
                                f"**Actual OC (2022):** {actual_oc:.1f}  |  **Abs error:** {abs(actual_oc - pred_oc):.1f}"
                            )
                            if actual_inv is not None:
                                reply += f"\n\n**Actual invoice_count (2022):** {actual_inv:.1f}"

                except Exception as e:
                    reply = f"Prediction error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# Tab 2: Forecast Mode
with tab_forecast:
    st.subheader("Forecast Mode (Future prediction)")
    st.caption("Enter forecast weather. App predicts invoice_count + OC split, enforces relationships, and adds OC 95% range.")

    store_id = st.selectbox("Store ID", ALL_STORES, index=0)
    date_val = st.date_input("Future date", value=pd.to_datetime("2026-02-25"))

    col1, col2 = st.columns(2)
    with col1:
        tmin = st.number_input("Forecast tmin (°C)", value=0.0)
        tmax = st.number_input("Forecast tmax (°C)", value=10.0)
        tavg = st.number_input("Forecast tavg (°C)", value=5.0)
    with col2:
        prcp = st.number_input("Forecast precipitation prcp (mm)", value=0.0)
        snow = st.number_input("Forecast snow (cm)", value=0.0)
        wspd = st.number_input("Forecast wind speed wspd (km/h)", value=10.0)

    if st.button("Predict from Forecast"):
        sid = int(store_id)
        d = pd.to_datetime(str(date_val)).normalize()

        X = build_forecast_features(sid, d, tmin, tmax, tavg, prcp, wspd, snow)
        out = predict_all(X)

        pred_inv = out["invoice_pred"]
        pred_nf = out["non_fleet_oc_pred"]
        pred_fl = out["fleet_oc_pred"]
        pred_oc = out["oc_total_pred"]

        sev = severity_bucket(tmin, tmax, prcp, snow)

        baseline_oc = get_baseline(BASELINE_OC, sid, d, fallback=float(demo["oc_count"].mean()))
        pchg = pct_change(pred_oc, baseline_oc)
        direction = "fewer" if pchg < 0 else "more"

        # 95% OC interval using severity-specific residual bounds from 2022
        qlo, qhi = BOUNDS.get(sev, (GLOBAL_LO, GLOBAL_HI))
        lo95 = max(pred_oc + qlo, 0.0)
        hi95 = max(pred_oc + qhi, 0.0)

        st.success(
            f"Based on forecast {format_sev(sev)} on **{d.date()}**, store **{sid}** should see about "
            f"**{abs(pchg):.1f}% {direction} oil-change visits** than a typical {d.strftime('%A')} in {d.strftime('%B')}.\n\n"
            f"**Prediction (invoice_count):** **{pred_inv:.1f}**\n\n"
            f"**Prediction (OC total):** **{pred_oc:.1f}**  (Non-fleet: {pred_nf:.1f} | Fleet: {pred_fl:.1f})\n\n"
            f"**Typical OC 95% range:** {lo95:.1f}–{hi95:.1f}"
        )

    st.markdown("---")
    st.caption("Tip: If chat errors, start Ollama with `ollama serve`.")
