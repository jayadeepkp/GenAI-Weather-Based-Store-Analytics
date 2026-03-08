# tests/test_sprint2_pipeline.py
# Run: pytest -q
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


# ----------------------------
# Helpers
# ----------------------------
def _must_exist(path: Path):
    assert path.exists(), f"Missing: {path}"


def safe_mape(y_true, y_pred, min_true: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= min_true
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _load_csv_any(paths: list[Path]) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    raise AssertionError(f"None of these files exist: {paths}")


# ----------------------------
# 1) Feature Upgrade v2 (artifacts exist)
# ----------------------------
def test_feature_upgrade_v2_artifacts_exist():
    # Weather cache (all stores)
    _must_exist(DP / "weather_allstores.parquet")

    # At least one model artifact should exist (invoice prototype or original)
    any_model = any([
        (DP / "model_hgb_allstores.joblib").exists(),
        (MODELS / "model_hgb_invoice.joblib").exists(),
        (MODELS / "model_hgb_non_fleet_oc.joblib").exists(),
        (MODELS / "model_hgb_fleet_oc.joblib").exists(),
    ])
    assert any_model, "No model artifacts found in data_processed/ or models/."

    # At least one feature list exists
    any_feat = any([
        (DP / "feature_cols_hgb_allstores.json").exists(),
        (REPORTS / "metrics_hgb_hierarchical.json").exists(),  # contains feature_cols
    ])
    assert any_feat, "No feature list/metrics artifact found."


# ----------------------------
# 2) 95% prediction interval (exists + valid)
# ----------------------------
def test_prediction_interval_columns_and_validity():
    # Hierarchical output should exist
    pred_path = DP / "predictions_hgb_hierarchical_2022.csv"
    _must_exist(pred_path)
    df = pd.read_csv(pred_path)

    # Interval columns may be named oc_lower_95/oc_upper_95 (your sprint2 notebook)
    lower_candidates = [c for c in ["oc_lower_95", "lower_95", "lower95"] if c in df.columns]
    upper_candidates = [c for c in ["oc_upper_95", "upper_95", "upper95"] if c in df.columns]
    assert lower_candidates and upper_candidates, f"Missing interval cols. Columns: {list(df.columns)}"

    lo = lower_candidates[0]
    hi = upper_candidates[0]

    # Interval sanity
    assert (df[lo].fillna(0) >= 0).all(), "Interval lower bound has negative values."
    assert (df[hi].fillna(0) >= 0).all(), "Interval upper bound has negative values."
    assert (df[lo] <= df[hi]).all(), "Found rows where lower_95 > upper_95."

    # Coverage check (not strict to avoid flakiness, but should be reasonable)
    if "oc_count" in df.columns:
        inside = ((df["oc_count"] >= df[lo]) & (df["oc_count"] <= df[hi])).mean()
        assert 0.85 <= inside <= 0.99, f"Interval coverage out of expected range: {inside:.3f}"


# ----------------------------
# 3) Hierarchical Modeling constraints
# ----------------------------
def test_hierarchical_constraints_hold():
    pred_path = DP / "predictions_hgb_hierarchical_2022.csv"
    _must_exist(pred_path)
    df = pd.read_csv(pred_path)

    # Required columns (allow minor naming differences)
    required = {
        "store_id", "invoice_date",
        "invoice_count", "oc_count", "fleet_oc_count",
        "pred_invoice", "pred_oc_total", "pred_fleet_oc", "pred_non_fleet_oc",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns for hierarchical constraints: {missing}"

    # Constraints
    assert (df["pred_oc_total"] <= df["pred_invoice"] + 1e-6).all(), "Constraint violated: OC > invoices"
    assert (df["pred_fleet_oc"] <= df["pred_oc_total"] + 1e-6).all(), "Constraint violated: fleet > OC"
    assert (df["pred_non_fleet_oc"] >= -1e-6).all(), "Constraint violated: non_fleet < 0"
    assert (df["pred_fleet_oc"] >= -1e-6).all(), "Constraint violated: fleet < 0"
    assert (df["pred_oc_total"] >= -1e-6).all(), "Constraint violated: OC total < 0"

    # OC total should equal non_fleet + fleet (within tolerance)
    diff = (df["pred_non_fleet_oc"] + df["pred_fleet_oc"] - df["pred_oc_total"]).abs()
    assert diff.max() <= 1e-3, f"pred_non_fleet_oc + pred_fleet_oc != pred_oc_total (max diff {diff.max():.6f})"

    # Key uniqueness
    dup = df.duplicated(["store_id", "invoice_date"]).sum()
    assert dup == 0, f"Duplicate (store_id, invoice_date) rows found: {dup}"


# ----------------------------
# 4) Heuristic v2 artifacts + basic sanity
# ----------------------------
def test_heuristic_v2_outputs_exist_and_sane():
    # Multipliers JSON (either of these)
    any_json = [
        REPORTS / "heuristic_multipliers_oc_v2.json",
        REPORTS / "heuristic_multipliers_oc_logadd.json",
    ]
    assert any(p.exists() for p in any_json), f"Missing heuristic multipliers JSON. Looked for: {any_json}"

    # Predictions CSV (any known name)
    heur_paths = [
        DP / "predictions_heuristic_oc_2022.csv",
        DP / "predictions_heuristic_oc_2022_logadd.csv",
    ]
    heur = _load_csv_any(heur_paths)

    # Must have keys + prediction column + actual column
    assert "store_id" in heur.columns and "invoice_date" in heur.columns, "Heuristic file missing keys."
    pred_candidates = [c for c in ["oc_pred_heuristic_v2", "oc_pred_heuristic", "oc_pred_heuristic_logadd", "oc_pred"] if c in heur.columns]
    actual_candidates = [c for c in ["oc_actual", "oc_count"] if c in heur.columns]

    assert pred_candidates, f"Heuristic missing prediction column. Columns: {list(heur.columns)}"
    assert actual_candidates, f"Heuristic missing actual OC column. Columns: {list(heur.columns)}"

    pred_col = pred_candidates[0]
    act_col = actual_candidates[0]

    # Sanity
    assert heur[pred_col].isna().sum() == 0, "Heuristic predictions contain NaNs."
    assert (heur[pred_col] >= 0).all(), "Heuristic predictions contain negative values."

    # Metrics should be finite
    mae = float(np.mean(np.abs(heur[act_col].astype(float) - heur[pred_col].astype(float))))
    smape = safe_mape(heur[act_col], heur[pred_col])
    assert np.isfinite(mae), "Heuristic MAE is not finite."
    assert np.isfinite(smape), "Heuristic safeMAPE is not finite."


# ----------------------------
# 5) Heuristic vs HGB comparison table (makes heuristic task 100%)
# ----------------------------
def test_comparison_table_exists_and_schema():
    cmp_path = DP / "comparison_heuristic_vs_hgb_oc_2022.csv"
    _must_exist(cmp_path)
    df = pd.read_csv(cmp_path)

    required = {"store_id", "invoice_date", "oc_actual_final", "oc_pred_hgb", "oc_pred_heuristic"}
    missing = required - set(df.columns)
    assert not missing, f"Comparison table missing columns: {missing}"

    assert df["oc_pred_hgb"].isna().sum() == 0, "Comparison: HGB preds contain NaNs."
    assert df["oc_pred_heuristic"].isna().sum() == 0, "Comparison: Heuristic preds contain NaNs."

    # keys should be unique
    dup = df.duplicated(["store_id", "invoice_date"]).sum()
    assert dup == 0, f"Comparison table has duplicate keys: {dup}"


# ----------------------------
# 6) Quantify weather impact (percent change vs normal + 95% CI) output exists
# ----------------------------
def test_weather_impact_confidence_outputs_present():
    # If you saved a table, check it here. If not, check that the notebook output file exists.
    # Recommended filename to save from your final cell:
    # data_processed/weather_impact_ci_2022.csv
    cand = [
        DP / "weather_impact_ci_2022.csv",
        REPORTS / "weather_impact_ci_2022.csv",
    ]
    assert any(p.exists() for p in cand), (
        "Missing saved weather impact CI table. "
        "Save your final impact table to data_processed/weather_impact_ci_2022.csv "
        "or reports/weather_impact_ci_2022.csv."
    )


# ----------------------------
# 7) Weather impact & error hotspot analysis exists
# ----------------------------
def test_error_hotspot_outputs_present():
    # Optional but recommended: save these outputs
    # - reports/error_hotspots_2022.csv
    # - reports/severity_impact_table_2022.csv
    cand1 = REPORTS / "severity_impact_table_2022.csv"
    cand2 = REPORTS / "error_hotspots_2022.csv"

    assert cand1.exists() or cand2.exists(), (
        "Missing saved hotspot/impact outputs. "
        "Save at least one: reports/severity_impact_table_2022.csv OR reports/error_hotspots_2022.csv"
    )