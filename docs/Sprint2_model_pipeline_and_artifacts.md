## Sprint Overview

The goal of Sprint 2 was to transition the project from initial exploratory work into a structured forecasting pipeline capable of predicting daily store traffic and oil-change demand while incorporating weather information.

This sprint focused on three major system components:
	1.	Dataset engineering framework to unify all data sources and feature engineering.
	2.	Hierarchical forecasting models to predict store-level demand.
	3.	Model diagnostics and interpretability analysis to evaluate performance and quantify the impact of weather variables.

By the end of the sprint, the team implemented a complete modeling workflow including dataset construction, model training, prediction generation, error analysis, and feature importance evaluation.


## 1. System Architecture

The overall modeling workflow developed during this sprint is shown below.

Raw Data Sources
   |
   |-- Store performance data (2018–2022)
   |-- Store metadata
   |-- Historical weather data
   |
   v
DatasetBuilder (Feature Engineering Layer)
   |
   v
Training Pipeline (HGB Baseline Model)
   |
   v
Hierarchical Forecasting Pipeline
   |
   v
Prediction Outputs + Prediction Intervals
   |
   v
Model Diagnostics & Feature Importance Analysis

This architecture separates the system into three core layers:
	1.	Dataset construction
	2.	Model training and prediction
	3.	Model evaluation and interpretation


## 2. Dataset Engineering Framework (Dylan)

A central contribution of this sprint was the implementation of a DatasetBuilder module, which standardizes how the modeling dataset is created.

The DatasetBuilder automates the entire feature engineering pipeline and ensures that all models use the same consistent dataset.

Data Sources

The builder integrates multiple data sources:

Store Performance Data
	•	Daily store transactions
	•	Invoice counts
	•	Oil-change counts

Store Metadata
	•	Store location
	•	Market information
	•	Operational characteristics

Weather Data
	•	Daily weather observations retrieved using Meteostat
	•	Weather severity classifications


Feature Engineering Pipeline

The DatasetBuilder constructs a modeling dataset by sequentially adding multiple types of features.

## 1. Store Features

Operational characteristics of each store:
	•	market_id
	•	store_state
	•	area_id
	•	marketing_area_id
	•	bay_count
	•	capacity_pressure

These variables capture structural differences between locations.


## 2. Calendar Features

Calendar-based variables capture recurring demand patterns.

Examples include:
	•	dow (day of week)
	•	month
	•	day_of_year
	•	is_weekend

These features model seasonal and weekly demand behavior.


## 3. Weather Features

Daily weather conditions are incorporated into the dataset using Meteostat.

Base weather variables include:
	•	tmin
	•	tmax
	•	tavg
	•	rain_mm
	•	snow_cm
	•	wspd
	•	temp_range

These variables allow the model to capture weather-driven demand changes.


## 4. Weather Severity Classification

To capture nonlinear weather effects, raw weather variables are transformed into severity categories:

Examples:
	•	sev_normal
	•	sev_freezing
	•	sev_snow_heavy
	•	sev_rain_heavy
	•	sev_cold_extreme
	•	sev_heat_extreme

These categorical features help the model learn behavioral patterns related to extreme weather events.


## 5. Weather Interaction Features

Additional derived features represent extreme weather conditions:

Examples include:
	•	heavy_rain
	•	heavy_snow
	•	extreme_heat
	•	extreme_cold
	•	snow_freezing

These variables capture more complex weather impacts.


## 6. Demand Lag Features

Lag features represent recent demand history.

Examples:
	•	invoice_count_lag_1
	•	invoice_count_lag_7
	•	invoice_count_lag_14

For oil-change demand:
	•	non_fleet_oc_lag_1
	•	fleet_oc_count_lag_7

These features capture demand persistence.


## 7. Rolling Demand Averages

Rolling averages smooth short-term fluctuations.

Examples:
	•	invoice_count_rollmean_7
	•	invoice_count_rollmean_28
	•	non_fleet_oc_rollmean_7
	•	fleet_oc_count_rollmean_28

These variables represent short-term demand trends.


## Final Dataset

After feature engineering, the dataset contains:
	•	67 model features
	•	Daily observations for each store
	•	Historical coverage from 2018–2022


## 3. Time-Based Data Split

To maintain realistic forecasting evaluation, the dataset is split chronologically.

Training period:

2018-01-01 → 2021-12-31

Validation period:

2022-01-01 → 2022-12-31

This prevents future data leakage and simulates real operational forecasting conditions.


## 4. Production Training Pipeline 

A production training script was implemented to train the baseline forecasting model using the DatasetBuilder output.

The pipeline performs the following steps:
	1.	Load dataset from DatasetBuilder
	2.	Extract feature columns
	3.	Apply preprocessing pipeline
	4.	Train baseline model
	5.	Evaluate validation performance
	6.	Save model artifacts


Model Used

The baseline model selected for the sprint is:

HistGradientBoostingRegressor (HGB)

This model was chosen because it:
	•	handles large datasets efficiently
	•	models nonlinear feature relationships
	•	supports mixed numeric and categorical features


Preprocessing Pipeline

The training pipeline includes:

Categorical Features
	•	encoded using One-Hot Encoding

Numeric Features
	•	missing values filled using median imputation

This preprocessing pipeline ensures the model receives clean numeric inputs.


Model Outputs

For each target variable the pipeline produces:
	•	trained model artifact
	•	validation predictions
	•	performance metrics
	•	feature column metadata

Saved artifacts include:

model.joblib
metrics.json
predictions_valid.csv
feature_cols.json


## 5. Hierarchical Forecasting Model (Jayadeep)

The hierarchical forecasting approach extends the baseline model by predicting multiple related targets.

The system predicts:
	•	invoice traffic
	•	non-fleet oil-change demand
	•	fleet oil-change demand

These predictions are combined to generate total oil-change forecasts.


Prediction Outputs

The hierarchical pipeline generates prediction datasets containing:
	•	actual values
	•	predicted values
	•	absolute errors
	•	prediction intervals

Example output columns:
	•	pred_invoice
	•	pred_non_fleet_oc
	•	pred_fleet_oc
	•	pred_oc_total
	•	oc_lower_95
	•	oc_upper_95

Prediction intervals provide uncertainty estimates around forecasts.



## 6. Model Diagnostics and Error Analysis (Harshini)

Model performance was analyzed using validation predictions.

The following diagnostic analyses were performed:

## Error by Store

Identifies locations where the model performs poorly.

This analysis helps detect:
	•	stores with unusual demand patterns
	•	operational anomalies
	•	missing store-specific features

## Error by Month

Evaluates seasonal performance.

This analysis reveals whether model accuracy varies during different periods of the year.

## Error by Volume Bucket

Stores were grouped by average demand volume.

Results showed that:
	•	higher-volume stores tend to have larger absolute errors
	•	relative accuracy remains stable across store groups

## Prediction Bias Analysis

Model bias was evaluated by comparing:
	•	overpredictions
	•	underpredictions

Results indicate a slight overall underprediction tendency.


## 7. Feature Importance Analysis (Harshini)

Permutation feature importance was used to determine which variables most influence predictions.

Most Important Predictors

For invoice traffic:
	•	invoice_count_rollmean_7
	•	dow
	•	invoice_count_rollmean_28
	•	invoice_count_lag_1
	•	day_of_year

For oil-change demand:
	•	non_fleet_oc_rollmean_7
	•	dow
	•	non_fleet_oc_rollmean_28
	•	non_fleet_oc_lag_1
	•	day_of_year

These results show that recent demand history and calendar patterns are the strongest predictors.

## 8. Weather Contribution Analysis

To evaluate the impact of weather features, two models were compared:
	1.	Full model with all features
	2.	Model without weather features

Model Comparison Results

Target	Full Model MAE	No Weather MAE	Impact
invoice_count	6.426	6.530	+0.105
oc_count	5.700	5.824	+0.124

Removing weather features consistently increased prediction error.

This confirms that weather improves model accuracy, even though it is not the dominant signal.


## 9. Weather Severity Impact Analysis

Demand was analyzed across weather severity categories.

Key observations:
	•	Extreme cold reduces invoice traffic by ~4.7 visits per day
	•	Heavy rain reduces traffic by ~3.9 visits
	•	Heavy snow reduces traffic by ~3.1 visits
	•	Freezing conditions reduce traffic by ~1.7 visits

Extreme heat shows different behavior, with slightly higher invoice traffic but lower oil-change demand.

⸻

## 10. Key Insights from Sprint 2
	1.	Store traffic is primarily driven by recurring demand patterns and short-term demand memory.
	2.	Weather provides secondary but meaningful predictive value.
	3.	Severe cold, snow, and rain conditions generally reduce store visits.
	4.	The forecasting system performs consistently across the validation year.
	5.	The project now has a fully reproducible modeling pipeline.


## 11. Sprint Deliverables

The following components were completed during this sprint:
	•	DatasetBuilder feature engineering framework
	•	Production training pipeline
	•	Hierarchical forecasting model
	•	Prediction interval generation
	•	Model diagnostics and bias analysis
	•	Feature importance evaluation
	•	Weather impact validation


## 12. Next Steps

Future work will focus on:
	•	improving model accuracy
	•	refining weather interaction features
	•	incorporating additional operational data
	•	building a GenAI interface for forecasting queries