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
OLLAMA_MODEL = "llama3.1"  # change if needed (ollama list)

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"

MODEL_PATH = DATA_PROCESSED / "model_hgb_allstores.joblib"
FEATURES_PATH = DATA_PROCESSED / "feature_cols_hgb_allstores.json"
DEMO_TABLE = DATA_PROCESSED / "demo_features_2022.parquet"

CHAT_SYSTEM = """
You are a helpful store operations assistant.
Speak in simple, clear language.
If the user asks for a prediction, you can provide a short explanation.
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

    # direct JSON
    try:
        return json.loads(content)
    except Exception:
        pass

    # fallback: extract {...}
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
# ML helpers
# ----------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features: {FEATURES_PATH}")
    if not DEMO_TABLE.exists():
        raise FileNotFoundError(f"Missing demo table: {DEMO_TABLE}")

    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    demo = pd.read_parquet(DEMO_TABLE)
    demo["invoice_date"] = pd.to_datetime(demo["invoice_date"]).dt.normalize()
    demo["store_id"] = demo["store_id"].astype(int)

    return model, feature_cols, demo

def predict_from_row(model, feature_cols, row: pd.DataFrame) -> dict:
    X = row[feature_cols].copy()
    y_pred = float(np.clip(model.predict(X)[0], 0, None))
    y_true = float(row["invoice_count"].iloc[0])

    abs_err = abs(y_true - y_pred)
    pct_err = (abs_err / y_true * 100.0) if y_true >= 30 else None  # avoid scary % on tiny days

    # Weather context
    tags = []
    if "is_rain" in row.columns and int(row["is_rain"].iloc[0]) == 1: tags.append("rain")
    if "is_snow" in row.columns and int(row["is_snow"].iloc[0]) == 1: tags.append("snow")
    if "is_freezing" in row.columns and int(row["is_freezing"].iloc[0]) == 1: tags.append("freezing temps")
    if not tags: tags.append("normal weather")

    return {
        "prediction": y_pred,
        "actual": y_true,
        "abs_error": abs_err,
        "pct_error": pct_err,
        "weather_tags": tags,
        "tmin": float(row["tmin"].iloc[0]) if "tmin" in row.columns else None,
        "prcp": float(row["prcp"].iloc[0]) if "prcp" in row.columns else None,
        "snow": float(row["snow"].iloc[0]) if "snow" in row.columns else None,
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Store Manager Assistant", page_icon="🛠️", layout="centered")
st.title("🛠️ Store Manager Assistant (Ollama + ML Prototype)")
st.caption("Chat normally. Ask for predictions like: **Predict store 79609 on 2022-01-06**")

model, feature_cols, demo = load_artifacts()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about store traffic or request a prediction for a 2022 store/date."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Type your message...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Decide mode
    if not looks_like_prediction_request(user_text):
        # normal chat with Ollama
        try:
            reply = ollama_chat(user_text)
        except Exception as e:
            reply = f"Error talking to Ollama: {e}"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # prediction mode
        req = ollama_extract(user_text)

        if req.get("error") == "missing_store_id":
            reply = "Please provide a store_id (example: **Predict store 79609 on 2022-01-06**)."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        elif req.get("error") == "missing_invoice_date":
            reply = "Please provide a 2022 date like **YYYY-MM-DD** (example: **2022-01-06**)."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        elif "error" in req:
            reply = "I couldn't extract store/date. Try: **Predict store 79609 on 2022-01-06**."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            try:
                sid = int(req["store_id"])
                d = pd.to_datetime(req["invoice_date"]).normalize()

                row = demo[(demo["store_id"] == sid) & (demo["invoice_date"] == d)]
                if row.empty:
                    available = (demo[demo["store_id"] == sid]["invoice_date"]
                                 .drop_duplicates().sort_values())
                    hint = available.head(5).dt.strftime("%Y-%m-%d").tolist()
                    reply = f"No data for store **{sid}** on **{d.date()}**. Try one of these dates: {hint}"
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)
                else:
                    out = predict_from_row(model, feature_cols, row)

                    reply = (
                        f"**Prediction:** store **{sid}** on **{d.date()}** → **{out['prediction']:.1f} invoices**\n\n"
                        f"**Actual (2022):** {out['actual']:.1f}\n\n"
                        f"**Abs error:** {out['abs_error']:.1f}"
                    )
                    if out["pct_error"] is not None:
                        reply += f" (**{out['pct_error']:.1f}%**)"
                    reply += "\n\n"
                    reply += f"**Weather context:** {', '.join(out['weather_tags'])} (tmin={out['tmin']}, prcp={out['prcp']}, snow={out['snow']})"

                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)

            except Exception as e:
                reply = f"Prediction error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)