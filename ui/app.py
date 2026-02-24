import streamlit as st
from datetime import date
import pandas as pd
import requests

# ---------- Page config ----------
st.set_page_config(
    page_title="Weather-Based Store Analytics",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .kpi-card {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
      }
      .kpi-title { font-size: 14px; opacity: 0.7; margin-bottom: 8px; }
      .kpi-value { font-size: 48px; font-weight: 750; line-height: 1.1; }
      .kpi-sub { margin-top: 10px; font-size: 13px; opacity: 0.7; }
      .section-title { font-size: 18px; font-weight: 750; margin: 6px 0 8px; }
      .muted { opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Header ----------
st.title("Weather-Based Store Analytics")
st.caption(
    "Select a mode, choose stores/date, and generate predictions. (Model + metastack will be connected next.)"
)

# ---------- Mock store list (TEMP) ----------
# Later: replace this with stores pulled from metastack (unique store_id values).
ALL_STORES = [f"Store_{i}" for i in range(1, 301)]  # 300 stores as example

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")

    mode = st.radio("Mode", ["Single Store", "Multi-Store", "Metrics"], index=0)

    st.divider()

    selected_date = st.date_input("Date", value=date.today())

    if mode == "Single Store":
        store_id = st.selectbox("Store", ALL_STORES, index=0)
        selected_stores = [store_id]  # for consistent handling
    elif mode == "Multi-Store":
        selected_stores = st.multiselect(
            "Select Stores (no limit)",
            options=ALL_STORES,
            default=[ALL_STORES[0], ALL_STORES[1], ALL_STORES[2]],
        )
        st.caption("Tip: type to search stores quickly.")
    else:
        selected_stores = []

    st.divider()
    run_btn = st.button("Run", use_container_width=True)

# ---------- Tabs ----------
tab_single, tab_multi, tab_metrics = st.tabs(["Single Store", "Multi-Store", "Metrics"])

# ---------------- Single Store ----------------
with tab_single:
    st.markdown(
        '<div class="section-title">Single Store Prediction</div>',
        unsafe_allow_html=True,
    )
    st.write("Pick one store + date and generate a prediction.")

    if mode != "Single Store":
        st.info("Switch Mode → Single Store (left sidebar) to run single-store prediction.")
        pred = "—"
        store_label = "—"
    else:
        store_label = selected_stores[0] if selected_stores else "—"
        if run_btn:
            # Placeholder prediction (replace with model prediction later)
            pred = 124
            st.success(f"Generated prediction for **{store_label}** on **{selected_date}**.")
        else:
            pred = "—"

    kpi_card(
        "Predicted invoice_count",
        str(pred),
        sub=f"Store: {store_label} • Date: {selected_date}",
    )

# ---------------- Multi Store ----------------
with tab_multi:
    st.markdown(
        '<div class="section-title">Multi-Store Predictions + Export</div>',
        unsafe_allow_html=True,
    )
    st.write("Pick any number of stores (no limit) + date, then generate a table and download CSV.")

    if mode != "Multi-Store":
        st.info("Switch Mode → Multi-Store (left sidebar) to generate multi-store predictions.")
    else:
        if not selected_stores:
            st.warning("Select at least 1 store in the sidebar.")
            df = pd.DataFrame(columns=["store_id", "date", "predicted_invoice_count"])
        else:
            if run_btn:
                # Placeholder predictions for selected stores
                df = pd.DataFrame(
                    {
                        "store_id": selected_stores,
                        "date": [selected_date] * len(selected_stores),
                        "predicted_invoice_count": [120 + (i % 15) for i in range(len(selected_stores))],
                    }
                )
                st.success(f"Generated table for **{len(selected_stores)}** stores on **{selected_date}**.")
            else:
                # Show an empty table until they click Run
                df = pd.DataFrame(columns=["store_id", "date", "predicted_invoice_count"])

        st.dataframe(df, use_container_width=True)

        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"multi_store_predictions_{selected_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------- Metrics ----------------
with tab_metrics:
    st.markdown(
        '<div class="section-title">Metrics (Baseline vs Model)</div>',
        unsafe_allow_html=True,
    )
    st.write("These will come from your metrics.json/report once connected.")

    c1, c2 = st.columns(2)

    with c1:
        kpi_card("Baseline MAE", "—", sub="Connect metrics.json later")
        kpi_card("Baseline MAPE", "—", sub="Connect metrics.json later")

    with c2:
        kpi_card("Model MAE", "—", sub="Connect metrics.json later")
        kpi_card("Model MAPE", "—", sub="Connect metrics.json later")

    st.info("Next step: we’ll load metrics automatically from /reports/metrics.json.")

# =========================
# Chatbot (bottom section)
# =========================
st.divider()
st.markdown('<div class="section-title">Store Assistant</div>', unsafe_allow_html=True)
st.caption("Ask about the demo, what each tab means, and what to connect next. (Optional: connect to Ollama.)")

# --- Optional: Ollama config ---
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"  # change if your ollama model name differs

CHAT_SYSTEM = """
You are a helpful assistant for a Weather-Based Store Analytics demo.
Speak in simple, clear language.
Give short practical answers.
If asked about predictions/metrics, explain what the app is showing (do not reveal hidden internal evaluation metrics).
"""

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
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def local_fallback_reply(user_text: str) -> str:
    t = user_text.lower()

    if "how" in t and ("use" in t or "run" in t):
        return (
            "Use the left sidebar:\n"
            "- Choose a mode (Single Store / Multi-Store / Metrics)\n"
            "- Pick date + store(s)\n"
            "- Click Run\n"
            "Then check the corresponding tab."
        )

    if "metrics" in t or "mae" in t or "mape" in t:
        return (
            "Metrics compare baseline vs model performance.\n"
            "MAE = average absolute error.\n"
            "MAPE = average percent error.\n"
            "Once metrics.json is connected, this section will show real values."
        )

    if "single" in t:
        return "Single Store mode predicts invoice_count for one chosen store + date."

    if "multi" in t or "csv" in t or "export" in t:
        return "Multi-Store mode builds a table for many stores and lets you download it as CSV."

    if "model" in t or "joblib" in t:
        return "Next step is connecting your saved model file (e.g., model.joblib) and using it instead of placeholder predictions."

    return (
        "I can help with:\n"
        "- how to run the demo\n"
        "- what each tab means\n"
        "- what to connect next (model, metastack, metrics)\n"
        "Ask me something specific."
    )

# Session state for messages
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi! I’m your store assistant. Ask me about the demo, predictions, or metrics."}
    ]

# Render chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at the bottom
user_text = st.chat_input("Message the assistant...")
if user_text:
    st.session_state.chat_messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # try ollama first; fallback if not available
    try:
        reply = ollama_chat(user_text)
    except Exception:
        reply = local_fallback_reply(user_text)

    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)