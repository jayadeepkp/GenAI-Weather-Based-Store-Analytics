# Weather Module

## Overview

This module implements a reusable Meteostat-based weather ingestion and feature engineering pipeline for all active stores in the 2018–2022 performance dataset.

It provides:

- Historical daily weather retrieval per store (via latitude/longitude)
- Deterministic local caching to prevent redundant API calls
- Enforced daily date continuity (no missing date gaps)
- Leakage-safe feature engineering
- Model-ready export aligned on (store_id, invoice_date)


## Data Range

Weather data is aligned to the performance dataset date range:

- Start: 2018-01-02  
- End: 2022-12-31  
- 1825 days per store  

Total stores processed: 439  
Total rows generated: 801,175  


## Retrieval Pipeline

File: `meteostat_client.py`

For each store:

1. Pull daily weather using Meteostat `Point(lat, lon)`
2. Select relevant columns:
   - tavg, tmin, tmax
   - prcp
   - snow
   - wspd
3. Enforce full daily continuity across the entire date range
4. Cache result locally


## Caching Strategy

- Cache directory: `data_raw/weather_cache/`
- One file per store:
  - `store_id=<ID>_daily.csv`
- On subsequent runs:
  - If file exists → `[CACHE_HIT]`
  - No API call is made
- Cache integrity validated in notebook

This ensures reproducibility and performance stability.


## Feature Engineering

File: `features.py`

Features are generated per store with strict chronological ordering.

### Lag Features
- tavg_lag1, tavg_lag3, tavg_lag7
- wspd_lag1, wspd_lag3, wspd_lag7
- etc.

### Rolling Features (Trailing Only, No Leakage)
- tavg_rollmean_3
- tavg_rollmean_7
- prcp_rollsum_3
- prcp_rollsum_7

### Anomaly Feature
- tavg_anom_30
  - Current temperature minus trailing 30-day mean

### Calendar Features
- dow (day of week)
- month
- is_weekend

All rolling calculations are backward-looking to prevent train/validation leakage.


## Validation Evidence

The following checks were performed:

- Continuity check:
  - 1825 daily rows per store
  - 0 stores failing continuity
- Missingness summary on raw weather columns:
  - snow ~35%
  - prcp ~13%
  - temperature/wind ~2–3%
- Visual inspection:
  - Example daily temperature plot saved to:
    `reports/plots/example_store_tavg.png`
- Cache validation:
  - Verified `[CACHE_HIT]` behavior


## Output

Model-ready weather feature table:

`data_processed/weather_features_2018_2022.csv`

Schema snapshot:

`data_processed/weather_features_schema.txt`

Columns:
- store_id
- invoice_date
- raw weather variables
- engineered lag/rolling/anomaly features
- calendar features


## Known Limitations

- Some stores have limited Meteostat station coverage resulting in NaNs.
- 5 stores had empty raw pulls; continuity is still enforced with NaN weather values.
- No imputation is performed at this stage (left for modeling pipeline).


## Integration

This module outputs a clean weather feature table aligned by:

(store_id, invoice_date)

It is ready for downstream merge into the modeling DatasetBuilder.