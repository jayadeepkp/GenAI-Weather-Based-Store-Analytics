# tests/test_demo_streamlit_app.py
from pathlib import Path
import json
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
DEMO_DIR = ROOT / "demo"

MODEL_PATH = DATA_PROCESSED / "model_hgb_allstores.joblib"
FEATURES_PATH = DATA_PROCESSED / "feature_cols_hgb_allstores.json"
DEMO_TABLE = DATA_PROCESSED / "demo_features_2022.parquet"
STREAMLIT_APP = DEMO_DIR / "app.py"


def test_streamlit_app_file_exists():
    assert STREAMLIT_APP.exists(), f"Missing Streamlit app: {STREAMLIT_APP}"


def test_demo_artifacts_exist():
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert FEATURES_PATH.exists(), f"Missing features: {FEATURES_PATH}"
    assert DEMO_TABLE.exists(), f"Missing demo table: {DEMO_TABLE}"


def test_demo_table_schema_and_size():
    df = pd.read_parquet(DEMO_TABLE)
    required = {"store_id", "invoice_date", "invoice_count"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    assert df["store_id"].nunique() > 100, "Demo table should contain many stores"
    assert len(df) > 10000, "Demo table seems too small"


def test_model_loads_and_predicts_one_row():
    model = joblib.load(MODEL_PATH)
    feature_cols = json.loads(FEATURES_PATH.read_text())

    df = pd.read_parquet(DEMO_TABLE)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.normalize()
    df["store_id"] = df["store_id"].astype(int)

    # Find one row that has all required feature columns available
    available_cols = [c for c in feature_cols if c in df.columns]
    sample = df.dropna(subset=available_cols).head(1)
    assert len(sample) == 1, "Could not find a complete row for prediction"

    X = sample[feature_cols].copy()
    pred = model.predict(X)

    assert pred.shape == (1,)
    assert float(pred[0]) >= 0, "Prediction should be non-negative"