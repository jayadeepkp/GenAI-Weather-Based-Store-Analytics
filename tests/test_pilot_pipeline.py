# tests/test_pilot_pipeline.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data_processed"


MERGED_PILOT = DATA_PROCESSED / "merged_model_ready_pilot_meteostat_clean.csv"

TRAIN_PILOT = DATA_PROCESSED / "train_pilot.csv"
VALID_PILOT = DATA_PROCESSED / "valid_pilot.csv"
BASELINE_PILOT = DATA_PROCESSED / "baseline_predictions_pilot.csv"


def test_pilot_merged_exists():
    assert MERGED_PILOT.exists(), f"Missing: {MERGED_PILOT}"


def test_pilot_merged_required_columns():
    df = pd.read_csv(MERGED_PILOT)

    # Require core business columns
    required = {"store_id", "invoice_date", "invoice_count"}
    missing = required - set(df.columns)
    assert not missing, f"Missing core columns: {missing}"

    # Require at least these weather fields (tavg/snow may not exist depending on pull)
    weather_required = {"tmin", "tmax", "prcp", "wspd"}
    missing_weather = weather_required - set(df.columns)
    assert not missing_weather, f"Missing required weather columns: {missing_weather}"

    # Optional weather fields: allow if present
    # (tavg and snow are nice-to-have, not must-have)
    # print("Optional present:", [c for c in ["tavg","snow"] if c in df.columns])


def test_pilot_merge_no_duplicate_keys():
    df = pd.read_csv(MERGED_PILOT)
    dup = df.duplicated(["store_id", "invoice_date"]).sum()
    assert dup == 0, f"Found {dup} duplicate (store_id, invoice_date) rows"


def test_train_valid_files_exist():
    assert TRAIN_PILOT.exists(), f"Missing: {TRAIN_PILOT}"
    assert VALID_PILOT.exists(), f"Missing: {VALID_PILOT}"


def test_train_valid_schema_and_no_leakage():
    train_df = pd.read_csv(TRAIN_PILOT)
    valid_df = pd.read_csv(VALID_PILOT)

    for df, name in [(train_df, "train"), (valid_df, "valid")]:
        for col in ["store_id", "invoice_date", "invoice_count"]:
            assert col in df.columns, f"{name} missing column: {col}"

    train_df["invoice_date"] = pd.to_datetime(train_df["invoice_date"])
    valid_df["invoice_date"] = pd.to_datetime(valid_df["invoice_date"])

    assert train_df["invoice_date"].max() < valid_df["invoice_date"].min(), \
        "Time leakage: train max date is not before valid min date"


def test_baseline_predictions_exist_and_schema():
    assert BASELINE_PILOT.exists(), f"Missing: {BASELINE_PILOT}"
    df = pd.read_csv(BASELINE_PILOT)

    cols = set(df.columns)
    assert {"store_id", "invoice_date"}.issubset(cols), f"Missing keys in baseline file. Columns: {list(df.columns)}"

    # accept common prediction column names
    pred_cols = {"baseline_pred", "y_pred", "pred"}
    assert any(c in cols for c in pred_cols), f"No prediction column found. Columns: {list(df.columns)}"