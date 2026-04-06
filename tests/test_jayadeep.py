"""
Test Case — Jayadeep Kothapalli
Title: Store Manager Gets Accurate Weather-Based OC Prediction for Next Week
Severity: Critical

User Story: As a Valvoline store manager, I want to ask what to expect
next week so I can make staffing and scheduling decisions based on
real weather forecasts — with a confidence range I can plan around.

Instructions:
1. Start Ollama      : ollama serve
2. Start API         : cd demo && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
3. Run tests         : pytest tests/test_jayadeep.py -v
"""

import os
import pickle
import pytest
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════
BASE_URL   = 'http://localhost:8000'
MODEL_PATH = 'notebooks/valvoline_production/valvoline_models_production.pkl'
EVAL_PATH  = 'notebooks/valvoline_evaluation/valvoline_models_v2.pkl'
DATA_PATH  = 'notebooks/valvoline_production/processed_data.csv'
PRED_CSV   = 'notebooks/valvoline_evaluation/predictions_2022.csv'
DEMO_STORE = 79609
NEXT_WEEK  = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')


# ════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════

@pytest.fixture(scope='module')
def models():
    assert os.path.exists(MODEL_PATH), \
        f'Production model not found: {MODEL_PATH}'
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def processed_data():
    assert os.path.exists(DATA_PATH), \
        f'Processed data not found: {DATA_PATH}'
    return pd.read_csv(DATA_PATH, parse_dates=['invoice_date'])


@pytest.fixture(scope='module')
def predictions_2022():
    assert os.path.exists(PRED_CSV), \
        f'predictions_2022.csv not found. Run evaluation notebook first.'
    return pd.read_csv(PRED_CSV)


# ════════════════════════════════════════════════
# TASK 1 — ML MODEL ACCURACY & STABILITY
# ════════════════════════════════════════════════

class TestMLModelAccuracy:

    def test_production_model_exists(self, models):
        """Production model file must exist and load correctly."""
        assert models is not None

    def test_all_6_models_present(self, models):
        """All 6 trained models must be saved."""
        required = [
            'model_B_oc_regression',
            'model_Q05_lower_bound',
            'model_Q95_upper_bound',
            'model_FWD',
            'model_FWD_Q05',
            'model_FWD_Q95',
        ]
        for key in required:
            assert key in models, f'Missing model: {key}'

    def test_batch_model_has_98_features(self, models):
        """Batch model must have exactly 98 features."""
        assert len(models['features']) == 98, \
            f"Expected 98, got {len(models['features'])}"

    def test_forward_model_has_69_features(self, models):
        """Forward model must have 69 features — no lags."""
        assert len(models['forward_features']) == 69, \
            f"Expected 69, got {len(models['forward_features'])}"

    def test_forward_model_has_no_lag_features(self, models):
        """Forward model must not use lag features — blind forecast."""
        lag_cols = [
            f for f in models['forward_features']
            if 'lag' in f or 'roll' in f
        ]
        assert len(lag_cols) == 0, \
            f'Forward model has lag features: {lag_cols}'

    def test_train_end_date_is_2022(self, models):
        """Production model must be trained through end of 2022."""
        assert models['train_end'] == '2022-12-31', \
            f"Expected 2022-12-31, got {models['train_end']}"

    def test_label_encoders_present(self, models):
        """Label encoders must be saved for categorical features."""
        assert 'label_encoders' in models
        assert len(models['label_encoders']) > 0

    def test_model_predicts_positive_values(self, models):
        """Model predictions must never be negative OC counts."""
        model_FWD = models['model_FWD']
        FWD_FEATS = models['forward_features']

        row = pd.DataFrame([{f: 0 for f in FWD_FEATS}])
        row['dow']                = 1
        row['month']              = 4
        row['year']               = 2026
        row['store_dow_baseline'] = 45
        row['tavg']               = 15
        row['prcp']               = 0
        row['snow']               = 0
        row['wspd']               = 10

        pred = float(model_FWD.predict(row)[0])
        assert pred >= 0, \
            f'Model predicted negative OC: {pred}'

    def test_mae_within_acceptable_range(self, predictions_2022):
        """
        MAE computed from real 2022 holdout predictions.
        152,898 actual store-day predictions vs real OC counts.
        """
        mae = np.mean(
            np.abs(
                predictions_2022['actual_oc'] -
                predictions_2022['predicted_oc']
            )
        )
        assert mae <= 6.0, \
            f'MAE {mae:.2f} exceeds threshold 6.0'
        print(f'\n   Real MAE from 2022 holdout: {mae:.2f}')

    def test_p95_improved_from_baseline(self, predictions_2022):
        """
        P95 computed from real 2022 holdout predictions.
        Must be lower than pre-outlier-reduction baseline of 13.8.
        """
        abs_errors = np.abs(
            predictions_2022['actual_oc'] -
            predictions_2022['predicted_oc']
        )
        p95_actual = np.percentile(abs_errors, 95)
        p95_before = 13.8

        assert p95_actual < p95_before, \
            f'P95 not improved: {p95_actual:.2f} vs baseline {p95_before}'
        print(f'\n   Real P95 from 2022 holdout: {p95_actual:.2f}')

    def test_interval_coverage_near_90_percent(self, predictions_2022):
        """
        Coverage computed from real 2022 holdout predictions.
        90% interval must cover 85-95% of actual OC counts.
        """
        coverage = np.mean(
            (predictions_2022['actual_oc'] >= predictions_2022['lower_90']) &
            (predictions_2022['actual_oc'] <= predictions_2022['upper_90'])
        )
        assert 0.85 <= coverage <= 0.95, \
            f'Coverage {coverage:.3f} outside 85-95% range'
        print(f'\n   Real coverage from 2022 holdout: {coverage:.3f}')

    def test_holiday_flags_in_features(self, models):
        """Holiday flags must be in model for outlier reduction."""
        holiday_flags = [
            'is_holiday', 'is_thanksgiving_week',
            'is_christmas_week', 'is_newyear_week',
            'is_july4_week', 'is_day_before_holiday',
            'is_day_after_holiday',
        ]
        for flag in holiday_flags:
            assert flag in models['features'], \
                f'Holiday flag missing: {flag}'

    def test_p_abnormal_in_features(self, models):
        """p_abnormal must be in features for outlier reduction."""
        assert 'p_abnormal' in models['features'], \
            'p_abnormal missing — outlier reduction not applied'

    def test_store_sensitivity_features_present(self, models):
        """Store-specific sensitivity features must be in model."""
        sensitivity_features = [
            'store_rain_sensitivity',
            'store_snow_sensitivity',
            'store_dow_baseline',
            'store_fleet_dependency',
        ]
        for feat in sensitivity_features:
            assert feat in models['features'], \
                f'Sensitivity feature missing: {feat}'

    def test_predictions_cover_all_stores(self, predictions_2022):
        """
        2022 predictions must cover a large number of stores —
        proves model works at scale not just for one store.
        """
        n_stores = predictions_2022['store_id'].nunique()
        assert n_stores >= 400, \
            f'Only {n_stores} stores in predictions — expected 400+'
        print(f'\n   Stores in 2022 holdout: {n_stores}')

    def test_predictions_sample_size_sufficient(self, predictions_2022):
        """
        2022 holdout must have enough rows for statistical credibility.
        """
        n_rows = len(predictions_2022)
        assert n_rows >= 100000, \
            f'Only {n_rows:,} prediction rows — expected 100,000+'
        print(f'\n   Prediction rows: {n_rows:,}')


# ════════════════════════════════════════════════
# TASK 2 — FORECAST MODE + WEATHER API
# ════════════════════════════════════════════════

class TestForecastMode:

    def test_api_is_healthy(self):
        """API must be running with models loaded."""
        try:
            r = requests.get(f'{BASE_URL}/health', timeout=5)
        except requests.exceptions.ConnectionError:
            pytest.fail(
                'API not running. Start with: '
                'cd demo && uvicorn api:app --port 8000 --reload'
            )
        assert r.status_code == 200
        assert r.json()['status'] == 'ok'
        assert r.json()['models'] == 'loaded'

    def test_forecast_returns_7_days(self):
        """Week forecast must return exactly 7 consecutive days."""
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200, \
            f'Forecast failed: {r.text}'
        data = r.json()
        assert len(data['predictions']) == 7, \
            f"Expected 7 days, got {len(data['predictions'])}"

        dates = [
            datetime.strptime(p['date'], '%Y-%m-%d')
            for p in data['predictions']
        ]
        for i in range(1, len(dates)):
            diff = (dates[i] - dates[i-1]).days
            assert diff == 1, \
                f'Non-consecutive dates: {dates[i-1]} → {dates[i]}'

    def test_real_weather_fetched_for_store_coordinates(self):
        """
        Temperatures must vary across 7 days — proves
        real Open-Meteo data was fetched using store GPS
        coordinates, not mocked or hardcoded.
        """
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        temps = [p['temp_c'] for p in r.json()['predictions']]
        assert len(set(temps)) > 1, \
            'All temperatures identical — not real weather data'

    def test_predictions_are_positive_with_valid_range(self):
        """
        All OC predictions must be positive with valid
        confidence range: low <= predicted <= high.
        """
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        for p in r.json()['predictions']:
            assert p['predicted_oc'] > 0, \
                f"Zero/negative prediction on {p['date']}"
            assert p['range_low'] >= 0
            assert p['range_low']    <= p['predicted_oc'], \
                f"Range low > predicted on {p['date']}"
            assert p['predicted_oc'] <= p['range_high'], \
                f"Predicted > range high on {p['date']}"

    def test_confidence_range_is_meaningful_for_planning(self):
        """
        Range must be wide enough to be meaningful for
        staffing decisions — not a trivially narrow band.
        Client feedback: floor and ceiling for planning.
        """
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        for p in r.json()['predictions']:
            width = p['range_high'] - p['range_low']
            assert width >= 5, \
                f'Range too narrow on {p["date"]}: {width} OC ' \
                f'— not useful for staffing planning'

    def test_weather_data_included_in_response(self):
        """
        Real weather values must be returned so manager
        can see what conditions drove the prediction.
        """
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        for p in r.json()['predictions']:
            assert 'temp_c'     in p
            assert 'precip_mm'  in p
            assert 'wind_kmh'   in p
            assert 'weather'    in p
            assert 'pct_impact' in p

    def test_clear_days_have_zero_weather_impact(self):
        """
        Clear days must have 0% weather impact —
        validates weather classification is correct.
        """
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        for p in r.json()['predictions']:
            if 'Clear' in p['weather']:
                assert p['pct_impact'] == 0.0, \
                    f"Clear day has non-zero impact: {p['pct_impact']}"

    def test_weekly_total_is_realistic(self):
        """Weekly OC total must be realistic for this store."""
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        total = r.json()['weekly_total']
        assert 100 <= total <= 600, \
            f'Unrealistic weekly total: {total} OC'

    def test_forecast_works_for_multiple_stores(self):
        """
        Forecast must work for multiple stores —
        proves viability at scale across 439 stores.
        """
        for store_id in [79609, 84321, 84831]:
            r = requests.get(
                f'{BASE_URL}/predict/week/{store_id}/{NEXT_WEEK}',
                timeout=15
            )
            assert r.status_code == 200, \
                f'Forecast failed for store {store_id}'
            assert len(r.json()['predictions']) == 7, \
                f'Wrong number of days for store {store_id}'

    def test_store_specific_sensitivity_applied(self):
        """
        Different stores must have different rain sensitivity —
        proves store-specific data is used not network averages.
        """
        r1 = requests.get(f'{BASE_URL}/stores/79609', timeout=5)
        r2 = requests.get(f'{BASE_URL}/stores/84321', timeout=5)
        assert r1.status_code == 200
        assert r2.status_code == 200

        rain1 = r1.json()['rain_impact_pct']
        rain2 = r2.json()['rain_impact_pct']

        assert rain1 != rain2, \
            'Both stores have identical rain sensitivity — ' \
            'store-specific data not being applied'
        assert rain1 < 0, 'Store 79609 rain impact must be negative'
        assert rain2 < 0, 'Store 84321 rain impact must be negative'


# ════════════════════════════════════════════════
# INTEGRATION — FULL PIPELINE END TO END
# ════════════════════════════════════════════════

class TestFullPipelineIntegration:

    def test_model_to_api_to_prediction_full_pipeline(
        self, models, predictions_2022
    ):
        """
        Full end-to-end test:
        Trained model → Real 2022 holdout verified →
        API loads model → Real weather fetched →
        Store-specific prediction → Valid confidence range returned.

        This is the core project deliverable.
        """
        # Step 1 — Model exists and is loaded
        assert 'model_FWD' in models
        assert len(models['forward_features']) == 69

        # Step 2 — Real 2022 predictions verified
        mae = np.mean(
            np.abs(
                predictions_2022['actual_oc'] -
                predictions_2022['predicted_oc']
            )
        )
        assert mae <= 6.0, f'MAE too high: {mae:.2f}'

        coverage = np.mean(
            (predictions_2022['actual_oc'] >= predictions_2022['lower_90']) &
            (predictions_2022['actual_oc'] <= predictions_2022['upper_90'])
        )
        assert 0.85 <= coverage <= 0.95

        # Step 3 — API is healthy
        health = requests.get(f'{BASE_URL}/health', timeout=5)
        assert health.json()['status'] == 'ok'
        assert health.json()['models'] == 'loaded'

        # Step 4 — Real weather fetched for store location
        r = requests.get(
            f'{BASE_URL}/predict/week/{DEMO_STORE}/{NEXT_WEEK}',
            timeout=15
        )
        assert r.status_code == 200
        data = r.json()

        # Step 5 — Store info correct
        assert data['store_id'] == DEMO_STORE
        assert data['city']     == 'Lexington'
        assert data['state']    == 'KY'

        # Step 6 — 7 days with valid predictions
        assert len(data['predictions']) == 7
        for p in data['predictions']:
            assert p['predicted_oc']  > 0
            assert p['range_low']    <= p['predicted_oc']
            assert p['predicted_oc'] <= p['range_high']

        # Step 7 — Model MAE within range
        assert 4.0 <= data['model_mae'] <= 7.0

        print(f"\nFull pipeline verified end-to-end")
        print(f"   Store     : {data['city']}, {data['state']}")
        print(f"   Week start: {data['week_start']}")
        print(f"   Weekly OC : {data['weekly_total']}")
        print(f"   Model MAE : {mae:.2f} (real 2022 holdout)")
        print(f"   Coverage  : {coverage:.3f} (real 2022 holdout)")
        print(f"   Day 1     : {data['predictions'][0]['day']} → "
              f"{data['predictions'][0]['predicted_oc']} OC "
              f"[{data['predictions'][0]['range_low']}"
              f"–{data['predictions'][0]['range_high']}]")
