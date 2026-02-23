# Sprint 1 – Demo Scope  
Project: GenAI Weather-Based Store Analytics  
Owner: Rudwika Manne  
Sprint: Sprint 1  

---

## 1. Objective

The goal of this prototype is to demonstrate a working Streamlit application that predicts daily invoice_count for a selected store using date and weather features.

This demo validates model integration and UI functionality.

---

## 2. Inputs

### User Inputs
- store_id (dropdown)
- date (date picker)

### Derived Date Features
- day_of_week
- month
- weekend_flag

### Weather Inputs
- temperature
- rainfall
- humidity
- wind_speed

---

## 3. Outputs

- Predicted invoice_count
- Display of model evaluation metrics

---

## 4. Evaluation Metrics

- Baseline MAE
- Baseline MAPE
- Model MAE
- Model MAPE

---

## 5. Sprint 1 Deliverable

- Functional Streamlit app running locally
- Integrated trained model (.joblib)
- Stable UI with input validation
- Visible evaluation metrics
