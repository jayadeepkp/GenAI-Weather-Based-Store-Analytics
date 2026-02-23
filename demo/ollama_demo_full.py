import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"   # change to your installed model name (ollama list)

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

MODEL_PATH = DATA_PROCESSED / "model_hgb_allstores.joblib"
FEATURES_PATH = DATA_PROCESSED / "feature_cols_hgb_allstores.json"
DEMO_TABLE = DATA_PROCESSED / "demo_features_2022.parquet"

# System prompt for general assistant behavior (store-manager friendly)
CHAT_SYSTEM = """
You are a helpful store operations assistant.
Speak in simple, clear language.
If the user asks for a prediction, you should ask for store_id and date if missing.
Keep answers short and practical.
"""

# System prompt for STRICT extraction (only used when we detect prediction intent)
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
        "options": {"temperature": 0.3}
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
        "format": "json"
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()

    # Try direct JSON
    try:
        return json.loads(content)
    except Exception:
        pass

    # Fallback: extract {...}
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start:end+1])

    return {"error": "bad_json", "raw": content}

def looks_like_prediction_request(text: str) -> bool:
    t = text.lower()
    triggers = [
        "predict", "prediction", "forecast", "estimate",
        "how many", "expected", "traffic", "invoices", "invoice_count",
        "store", "2022-", "on 2022"
    ]
    return any(x in t for x in triggers)

# ----------------------------
# ML predictor helpers
# ----------------------------
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features: {FEATURES_PATH}")
    if not DEMO_TABLE.exists():
        raise FileNotFoundError(f"Missing demo table: {DEMO_TABLE} (create it from notebook)")

    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    demo = pd.read_parquet(DEMO_TABLE)
    demo["invoice_date"] = pd.to_datetime(demo["invoice_date"]).dt.normalize()
    demo["store_id"] = demo["store_id"].astype(int)

    return model, feature_cols, demo

def predict_from_row(model, feature_cols, row: pd.DataFrame) -> dict:
    X = row[feature_cols].copy()
    y_pred = float(np.clip(model.predict(X)[0], 0, None))
    y_true = float(row["invoice_count"].iloc[0]) if "invoice_count" in row.columns else None

    abs_err = None
    pct_err = None
    if y_true is not None:
        abs_err = abs(y_true - y_pred)
        if y_true >= 30:  # only show % when actual isn't tiny
            pct_err = abs_err / y_true * 100.0

    # Weather context (nice for “why”)
    tmin = row["tmin"].iloc[0] if "tmin" in row.columns else None
    prcp = row["prcp"].iloc[0] if "prcp" in row.columns else None
    snow = row["snow"].iloc[0] if "snow" in row.columns else None
    is_rain = int(row["is_rain"].iloc[0]) if "is_rain" in row.columns else 0
    is_freezing = int(row["is_freezing"].iloc[0]) if "is_freezing" in row.columns else 0
    is_snow = int(row["is_snow"].iloc[0]) if "is_snow" in row.columns else 0

    tags = []
    if is_rain: tags.append("rain")
    if is_snow: tags.append("snow")
    if is_freezing: tags.append("freezing temps")
    if not tags: tags.append("normal weather")

    return {
        "prediction": y_pred,
        "actual": y_true,
        "abs_error": abs_err,
        "pct_error": pct_err,
        "weather_tags": tags,
        "tmin": tmin,
        "prcp": prcp,
        "snow": snow
    }

# ----------------------------
# Main assistant loop
# ----------------------------
def main():
    model, feature_cols, demo = load_artifacts()

    print("Store Manager Assistant is ready")
    print("You can chat normally.")
    print("Prediction examples:")
    print("  Predict store 79609 on 2022-01-04")
    print("  Forecast store 84321 on 2022-12-24")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        # If it doesn't look like prediction intent, just chat normally
        if not looks_like_prediction_request(user):
            try:
                reply = ollama_chat(user)
                print("Assistant:", reply)
            except Exception as e:
                print("Assistant: Error talking to Ollama:", e)
            continue

        # Prediction intent: extract store_id/date, then predict
        req = ollama_extract(user)

        if req.get("error") == "missing_store_id":
            print("Assistant: Please tell me the store_id (example: 'Predict store 79609 on 2022-01-06').")
            continue
        if req.get("error") == "missing_invoice_date":
            print("Assistant: Please give a 2022 date like YYYY-MM-DD (example: '2022-01-06').")
            continue
        if "error" in req and req["error"] not in (None, ""):
            print("Assistant: I couldn't extract store/date from that. Try: 'Predict store 79609 on 2022-01-06'.")
            # optional debug: print(req)
            continue

        try:
            sid = int(req["store_id"])
            d = pd.to_datetime(req["invoice_date"]).normalize()
        except Exception:
            print("Assistant: Please use a valid store_id and a date like 2022-01-06.")
            continue

        row = demo[(demo["store_id"] == sid) & (demo["invoice_date"] == d)]
        if row.empty:
            available = (demo[demo["store_id"] == sid]["invoice_date"]
                         .drop_duplicates().sort_values())
            hint = available.head(5).dt.strftime("%Y-%m-%d").tolist()
            print(f"Assistant: I don't have data for store {sid} on {d.date()}. Try one of these dates: {hint}")
            continue

        out = predict_from_row(model, feature_cols, row)

        # Response (store-manager friendly)
        msg = f"Assistant: For store {sid} on {d.date()}, I predict about {out['prediction']:.1f} invoices."
        if out["actual"] is not None:
            msg += f" The actual was {out['actual']:.1f}."
            msg += f" Error: {out['abs_error']:.1f}"
            if out["pct_error"] is not None:
                msg += f" ({out['pct_error']:.1f}%)."
            else:
                msg += "."
        print(msg)

        # short “why” context
        tmin = out["tmin"]
        prcp = out["prcp"]
        snow = out["snow"]
        print(f"Assistant: Weather context: {', '.join(out['weather_tags'])} (tmin={tmin}, prcp={prcp}, snow={snow}).")

if __name__ == "__main__":
    main()