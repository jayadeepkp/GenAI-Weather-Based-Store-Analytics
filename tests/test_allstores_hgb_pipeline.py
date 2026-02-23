# tests/test_allstores_hgb_pipeline.py
from pathlib import Path
import json
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "reports"

WEATHER_FILE = DATA_PROCESSED / "weather_allstores.parquet"

MODEL_FILE = DATA_PROCESSED / "model_hgb_allstores.joblib"
FEATURES_FILE = DATA_PROCESSED / "feature_cols_hgb_allstores.json"
PRED_FILE = DATA_PROCESSED / "predictions_hgb_allstores.csv"
METRICS_FILE = REPORTS / "metrics_hgb_allstores.json"


def test_weather_allstores_file_exists():
    assert WEATHER_FILE.exists(), f"Missing: {WEATHER_FILE}"


def test_weather_allstores_schema():
    df = pd.read_parquet(WEATHER_FILE)
    required = {"store_id", "invoice_date", "tmin", "tmax", "prcp", "wspd"}
    missing = required - set(df.columns)
    assert not missing, f"Missing weather columns: {missing}"
    assert df["store_id"].nunique() > 100, "Weather file should contain many stores"
    assert len(df) > 10000, "Weather file seems too small"


def test_model_file_exists_and_loads():
    assert MODEL_FILE.exists(), f"Missing: {MODEL_FILE}"
    model = joblib.load(MODEL_FILE)
    assert model is not None, "Loaded model is None"


def test_feature_list_exists_and_not_empty():
    assert FEATURES_FILE.exists(), f"Missing: {FEATURES_FILE}"
    feats = json.loads(FEATURES_FILE.read_text())
    assert isinstance(feats, list) and len(feats) > 0, "Feature list is empty or invalid"
    assert "store_id" in feats, "store_id should be included in feature list"


def test_predictions_file_exists_and_schema():
    assert PRED_FILE.exists(), f"Missing: {PRED_FILE}"
    df = pd.read_csv(PRED_FILE)
    required = {"store_id", "invoice_date", "actual", "pred", "abs_error"}
    missing = required - set(df.columns)
    assert not missing, f"Missing prediction columns: {missing}"
    assert len(df) > 1000, "Predictions file seems too small"


def test_metrics_file_exists_and_has_keys():
    assert METRICS_FILE.exists(), f"Missing: {METRICS_FILE}"
    m = json.loads(METRICS_FILE.read_text())
    for k in ["model", "target", "mae", "safe_mape", "n_stores"]:
        assert k in m, f"Missing metrics key: {k}"