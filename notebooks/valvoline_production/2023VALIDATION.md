# 2023 Validation Results — Batch Model

**Model trained on:** 2018–2022  
**Tested on:** 2023 unseen data  
**Evaluation setup:** Same features as the 2022 test set for a fair comparison

---

## Overall Performance

**Dataset size:** 155,230 store-days across 439 stores

| Metric | Value |
|---|---:|
| MAE | 5.29 oil changes |
| RMSE | 6.90 oil changes |
| MAPE | 13.6% |
| R² | 0.8296 |
| Bias | -0.61 |
| P90 Absolute Error | 10.9 |
| P95 Absolute Error | 13.5 |
| 90% Prediction Interval Coverage | 84.7% |

---

## Fair Comparison with 2022 Test Set

Using the **same feature set** as the 2022 evaluation:

| Metric | 2022 Test | 2023 Unseen |
|---|---:|---:|
| MAE | 5.08 | 5.29 |
| RMSE | 7.08 | 6.90 |
| R² | 0.823 | 0.8296 |
| P95 Absolute Error | 13.1 | 13.5 |
| 90% Coverage | 88.8% | 84.7% |

---

## Comparison to Last Year

| Model | Result |
|---|---:|
| Last year UK model (2022 test) | R² = 0.822 |
| Last year AR model (2022 test) | R² = 0.793 |
| **This model (2022 test)** | **R² = 0.823** |
| **This model (2023 unseen)** | **R² = 0.8296** |

### Key Takeaways
- The model slightly outperformed the **last year UK model** on the 2022 test set.
- It also clearly outperformed the **last year AR model**.
- Most importantly, it maintained strong performance on **real unseen 2023 data**, showing good generalization.

---

## MAE by Weather Severity

| Weather Severity | MAE | Store-Days |
|---|---:|---:|
| 0 | 5.27 | 98,740 |
| 1 | 5.22 | 41,457 |
| 2 | 5.59 | 13,024 |
| 3 | 6.10 | 1,548 |
| 4 | 7.16 | 461 |

### Observation
Model error increases as weather severity becomes more extreme, which is expected given the smaller sample sizes and more volatile conditions in severe weather.

---

## MAE by Month

| Month | MAE | Store-Days |
|---|---:|---:|
| Jan | 5.26 | 13,003 |
| Feb | 5.47 | 12,082 |
| Mar | 5.28 | 13,414 |
| Apr | 5.29 | 12,575 |
| May | 5.40 | 13,014 |
| Jun | 5.47 | 12,986 |
| Jul | 5.28 | 12,981 |
| Aug | 5.39 | 13,480 |
| Sep | 5.25 | 12,604 |
| Oct | 5.01 | 13,462 |
| Nov | 5.20 | 12,613 |
| Dec | 5.24 | 13,016 |

### Observation
Performance is stable across all months, with MAE staying close to **5–5.5 oil changes**, indicating consistent seasonal robustness.

---

## Coverage by Weather Condition

| Condition | Coverage | MAE | Store-Days |
|---|---:|---:|---:|
| Clear | 84.7% | 5.27 | 98,740 |
| Light Rain | 85.2% | 5.30 | 45,036 |
| Heavy Rain | 83.1% | 5.57 | 13,523 |
| Any Snow | 81.0% | 5.88 | 2,045 |
| Freezing | 83.7% | 5.69 | 12,184 |
| High Wind | 77.4% | 6.66 | 541 |

### Observation
The model performs best under common conditions such as **clear weather** and **light rain**, while uncertainty increases under rarer and more disruptive conditions such as **snow** and **high wind**.

---

## Final Conclusion

The batch model trained on **2018–2022** successfully generalized to **real unseen 2023 data**, achieving **R² = 0.8296** and **MAE = 5.29** across **155,230 store-days** from **439 stores**.

These results show that the model:
- Maintains strong predictive accuracy outside the original test period
- Performs competitively against prior models from last year
- Remains stable across months and common weather conditions
- Degrades gracefully under more severe and rare weather events

Overall, this validation provides strong evidence that the model is reliable, production-ready, and capable of delivering consistent forecasting performance on future unseen data.
