============================================================
2023 VALIDATION RESULTS — BATCH MODEL
Model trained 2018-2022 | Tested on 2023 (unseen)
Same features as 2022 test — fair comparison
============================================================

OVERALL (155,230 store-days, 439 stores)
  MAE          : 5.29 oil changes
  RMSE         : 6.90 oil changes
  MAPE         : 13.6%
  R²           : 0.8296
  Bias         : -0.61
  P90 Abs Error: 10.9
  P95 Abs Error: 13.5
  90% Coverage : 84.7%

FAIR COMPARISON — SAME FEATURES:
  Metric               2022 test            2023 unseen
  ------------------------------------------------------------
  MAE                  5.08                 5.29
  RMSE                 7.08                 6.90
  R²                   0.823                0.8296
  P95                  13.1                 13.5
  90% Coverage         88.8%                84.7%

COMPARE TO LAST YEAR:
  Last year UK model (2022 test) : R² 0.822
  Last year AR model (2022 test) : R² 0.793
  Your model (2022 test)         : R² 0.823  beats last year
  Your model (2023 unseen)       : R² 0.8296  validated on real data

MAE BY WEATHER SEVERITY:
  Severity 0: MAE 5.27  (98,740 days)
  Severity 1: MAE 5.22  (41,457 days)
  Severity 2: MAE 5.59  (13,024 days)
  Severity 3: MAE 6.10  (1,548 days)
  Severity 4: MAE 7.16  (461 days)

MAE BY MONTH:
  Jan: MAE 5.26  (13,003 days)
  Feb: MAE 5.47  (12,082 days)
  Mar: MAE 5.28  (13,414 days)
  Apr: MAE 5.29  (12,575 days)
  May: MAE 5.40  (13,014 days)
  Jun: MAE 5.47  (12,986 days)
  Jul: MAE 5.28  (12,981 days)
  Aug: MAE 5.39  (13,480 days)
  Sep: MAE 5.25  (12,604 days)
  Oct: MAE 5.01  (13,462 days)
  Nov: MAE 5.20  (12,613 days)
  Dec: MAE 5.24  (13,016 days)

COVERAGE BY WEATHER CONDITION:
  Clear          : Coverage 84.7%  MAE 5.27  (98,740 days)
  Light Rain     : Coverage 85.2%  MAE 5.30  (45,036 days)
  Heavy Rain     : Coverage 83.1%  MAE 5.57  (13,523 days)
  Any Snow       : Coverage 81.0%  MAE 5.88  (2,045 days)
  Freezing       : Coverage 83.7%  MAE 5.69  (12,184 days)
  High Wind      : Coverage 77.4%  MAE 6.66  (541 days)

============================================================
CONCLUSION: Batch model trained 2018-2022
  R² 0.8296 on real unseen 2023 data
  MAE 5.29 on 155,230 real store-days
============================================================
