# Prototype 1 Notes — Weather-Based Store Traffic Prediction (Valvoline Retail)

## What Prototype 1 Does
Prototype 1 builds an end-to-end prediction pipeline that estimates **daily store traffic** using `invoice_count` as a proxy for visits. The model uses:
- **Historical store performance** (daily invoice counts per store)
- **Historical weather** at each store location (Meteostat)
- **Engineered features** (calendar + weather flags + lag/rolling demand signals)

The prototype is validated with a **time-based holdout** and supports a **GenAI demo** where a user can ask for store/date predictions in natural language.

---

## Data Used
**Customer data (raw):**
- `store_performance_2018to2022.csv`
  - `invoice_date`, `store_id`, `invoice_count`, `oc_count`, `fleet_oc_count`
- `store_info.csv`
  - includes `store_latitude`, `store_longitude`

**Weather data (generated):**
- `data_processed/weather_allstores.parquet`
  - pulled from Meteostat using store latitude/longitude
  - daily fields used include: `tmin`, `tmax`, `tavg`, `prcp`, `wspd`, `snow` (availability may vary by station)

**Demo lookup table (generated):**
- `data_processed/demo_features_2022.parquet`
  - model-ready feature rows for 2022 used by the Streamlit + Ollama demo

---

## Weather Pipeline Notes (Meteostat)
- Weather is pulled **store-by-store** to improve reliability and allow caching/retry.
- Output is saved as a single parquet file: `weather_allstores.parquet`.
- We only model stores with stable coverage over time (example run produced **434 stores** with weather coverage).
- Weather is joined to performance data using:
  - **Key:** `(store_id, invoice_date)`
  - `invoice_date` is normalized to day-level (`YYYY-MM-DD`).
- Data integrity checks used:
  - duplicate key checks (expected: 0 duplicate `(store_id, invoice_date)` pairs)
  - coverage checks (store count, date ranges, missing percentages)

---

## Merge + Integrity Behavior
**Merge method:**
- `perf` and `weather_all` merged on `(store_id, invoice_date)` with an inner join for the all-stores prototype.

**Duplicate handling:**
- If raw performance has duplicates on `(store_id, invoice_date)`, values are aggregated (sum) to enforce a single daily record per store.
- Weather duplicates are removed/aggregated to enforce one weather row per store/day.

**Observed behavior:**
- After cleaning, the merged dataset `m` contains a large multi-year panel with one row per store/day for all stores with weather coverage.

---

## Feature Engineering (Prototype 1)
Prototype 1 includes three main feature groups:

### 1) Calendar / Seasonality Features
Used to capture weekly and yearly patterns:
- `dow` (day of week)
- `month`
- `day_of_year`
- `year`
- `is_weekend`

### 2) Weather Signal + Intensity Features
Used to model weather effects more clearly than raw values alone:
- Raw: `tmin`, `tmax`, `tavg`, `prcp`, `wspd`, `snow`
- Derived:
  - `rain_mm`, `snow_cm`
  - `temp_range = tmax - tmin`
- Flags:
  - `is_rain` (prcp > 0)
  - `is_snow` (snow > 0)
  - `is_freezing` (tmin <= 0)
  - `heavy_rain` (rain_mm >= 10)
  - `heavy_snow` (snow_cm >= 5)
  - `severe_weather` (simple combined indicator for extreme cold/snow/rain)

### 3) Demand Memory (Lag / Rolling) Features
Used to capture store-level momentum and local demand shifts:
- Demand lags:
  - `inv_lag_1`, `inv_lag_7`, `inv_lag_14`
- Rolling means (computed using only past values):
  - `inv_roll7_mean`, `inv_roll14_mean`, `inv_roll28_mean`

### Leakage Prevention
All lag/rolling features are computed using `shift(1)` before rolling, ensuring each day’s features only use information available **up to the previous day**. This prevents future leakage within the validation year.

---

## Model Choice (Prototype 1)
**Model:** `HistGradientBoostingRegressor` inside an sklearn Pipeline  
**Preprocessing:**
- `store_id` encoded using OneHotEncoding
- numeric features median-imputed
- encoder configured to produce **dense output** (required for HGB)

**Why this model is a good prototype choice:**
- Handles non-linear relationships well (weather effects are often non-linear)
- Trains efficiently on large tabular data
- Works well with mixed categorical + numerical feature sets via pipeline

---

## Train / Validation Split
Prototype 1 uses a strict time-based holdout:
- **Train:** 2018–2021
- **Validate:** 2022

**Leakage check:**
- confirms `max(train_date) < min(valid_date)` is True

This split mirrors real deployment where we train on past years and evaluate on the next year.

---

## Metrics & How to Interpret
Prototype 1 reports:
- **MAE (Mean Absolute Error):** average absolute difference between predicted and actual invoices.
  - Interpretation: MAE ≈ 7–8 means the model is off by ~7–8 invoices per day on average.
- **safeMAPE:** percent error computed only when actual invoices are not tiny (helps avoid inflated percentages when actual is near zero).

Artifacts saved:
- `data_processed/predictions_hgb_allstores.csv` (row-level predicted vs actual)
- `reports/metrics_hgb_allstores.json` (summary metrics + run metadata)

---

## Model Behavior Observations
### What the model learns well
- Regular weekly patterns (weekday vs weekend)
- Seasonal changes (winter vs summer demand shifts)
- Some measurable weather impact (rain/freezing/snow affecting traffic)
- Store-to-store differences (store_id effect is strong)

### Where the model struggles
- **Outlier days** with unusually low or high invoices:
  - holidays, partial closures, staffing issues, local events
  - abrupt anomalies not explained by weather + calendar
- A few individual days can have large error even if overall MAE is good.
  - This is expected in real operations data.

### Why large single-day errors happen (important for demo)
If an example shows a big miss (e.g., actual 16 vs predicted 52):
- actual may be depressed by a non-weather factor (closure/holiday/event)
- the model’s features cannot observe closures unless we add a “holiday/closure” feature later
- overall evaluation should rely on aggregate metrics, not worst-case days only

---

## Demo Behavior (Streamlit + Ollama)
**Demo UI:** `demo/app.py` (Streamlit)  
**How it works:**
1. User types normal text (e.g., “Predict store 79609 on 2022-01-06”)
2. Ollama extracts `store_id` and `invoice_date` (JSON format)
3. App looks up the feature row from `demo_features_2022.parquet`
4. Model predicts `invoice_count`
5. Demo shows:
   - Prediction
   - Actual (2022)
   - Error
   - Weather context flags (rain/snow/freezing)

**Important note about “using 2022 data”:**
- The model is trained only on 2018–2021.
- For 2022 predictions, lag features use prior days’ invoices up to that date (realistic operational forecasting) and do not use future invoices.

---

## Testing / Verification
Automated tests are included and passing:
- pilot pipeline checks
- all-stores pipeline checks
- demo artifact + prediction sanity check

Run:
```bash
pytest -q
```
## Known Limitations (Prototype 1)
- No explicit holiday/closure/event indicators (likely source of outliers)
- No regional macro variables beyond store_id (economic, competition, promotions)
- Demo predicts using historical 2022 rows (not true future forecasting yet)
- Forecasting beyond dataset dates would require forecast weather + updated lag strategy