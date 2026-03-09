Model Pipeline, Diagnostics, and Artifact Management

Overview

This document explains the modeling pipeline implemented in this repository for the Weather-Based Store Traffic Prediction system.

The pipeline combines:
	•	dataset construction and feature engineering
	•	hierarchical HGB model training
	•	weather-aware demand modeling
	•	prediction interval generation
	•	model diagnostics and feature importance analysis
	•	a reusable inference wrapper for UI integration

The goal is to provide reproducible training, interpretable diagnostics, and a simple prediction interface for the UI layer.


1. Dataset Construction

The modeling dataset is built using:

src/dataset.py

This module uses the DatasetBuilder class to construct the training dataset by combining:
	•	store performance data
	•	store location metadata
	•	historical weather data
	•	engineered demand features

Feature engineering includes:

Demand Memory Features

Lag and rolling statistics capturing short-term demand momentum:

inv_lag_1
inv_lag_7
inv_rollmean_7
inv_rollmean_14
inv_rollmean_28

Weather Features

Weather variables and lagged weather signals:

tavg
tmin
tmax
prcp
snow
wspd
tavg_lag*
tmin_lag*
tmax_lag*

Calendar Features

dow
month

These features allow the model to learn baseline traffic patterns and weather-driven variation.


2. Model Training Pipeline

Training is executed using:

scripts/training_pipeline.py

This script trains a HistGradientBoostingRegressor (HGB) model.

Train / Validation Split

A strict time split is used:

Period	Purpose
2018–2021	Training
2022	Validation

This ensures evaluation reflects true forward prediction.

Running Training

Examples:

python scripts/training_pipeline.py --target invoice_count
python scripts/training_pipeline.py --target oc_count

Targets supported:
	•	invoice_count
	•	oc_count


3. Hierarchical Model and Interval System

Sprint 2 introduces a hierarchical modeling strategy implemented in:

07_sprint2_hierarchical_oc_weather_intervals.ipynb

Key additions include:

Hierarchical Model Structure

Instead of a single global model, hierarchical adjustments are applied to improve prediction accuracy across stores with different demand levels.

Heuristic Multipliers

Saved in:

reports/heuristic_multipliers_oc_v2.json

These adjustments help stabilize predictions for different traffic segments.

Prediction Intervals

95% prediction intervals are generated to provide uncertainty estimates.

Intervals allow predictions to be expressed as:

lower bound
prediction
upper bound

This improves decision-making for staffing and operational planning.


4. Model Diagnostics

Model performance diagnostics were performed in:

notebooks/model_diagnostics_and_feature_signal.ipynb

The diagnostics analyze:
	•	error by store
	•	error by month
	•	error by volume bucket
	•	overprediction vs underprediction bias
	•	worst prediction failures

Key Findings
	•	A small number of stores account for a large share of total error.
	•	December shows the highest prediction error, likely due to holiday demand effects.
	•	Very high-volume stores are harder to predict accurately.
	•	The model shows a mild underprediction bias during high-traffic periods.

These diagnostics highlight where future improvements should focus.


5. Feature Importance and Weather Signal Stability

Permutation importance analysis was used to evaluate feature contribution.

Two models were compared:

Model	Features
Full Model	Demand + Calendar + Weather
No-Weather Model	Demand + Calendar

Results

Target	Full MAE	No-Weather MAE	Difference
invoice_count	6.650	6.799	+0.149
oc_count	6.178	6.333	+0.155

Removing weather features degrades model accuracy, confirming that weather contributes meaningful predictive signal.

Interpretation

The system functions as:

baseline demand prediction with weather adjustment

Core predictive signals include:
	•	day of week
	•	demand memory features
	•	rolling demand averages

Weather provides secondary adjustments to baseline demand.


6. Model Inference Wrapper

A reusable inference wrapper was implemented in:

src/model/inference.py

This provides a simple interface for generating predictions.

Main Function

predict(store_id, start_date, end_date, target)

Example:

from src.model.inference import predict

df = predict(
    store_id=84321,
    start_date="2022-06-10",
    end_date="2022-06-15",
    target="invoice_count"
)

Output

The function returns:

store_id
invoice_date
yhat
yhat_lower
yhat_upper

What the Wrapper Handles

The wrapper automatically:
	1.	loads the trained model artifact
	2.	loads the saved feature schema
	3.	reconstructs the feature dataset using DatasetBuilder
	4.	handles feature compatibility with trained models
	5.	returns predictions with confidence intervals

This allows the UI to call the model without embedding ML logic inside the app.


7. Artifact Management

Training outputs are stored locally in:

artifacts/

Example structure:

artifacts/
  training_pipeline_invoice_v1/
      model.joblib
      feature_cols.json
      metrics.json
      predictions_valid.csv

  training_pipeline_oc_v1/
      model.joblib
      feature_cols.json
      metrics.json
      predictions_valid.csv

Artifacts are not committed to Git to avoid repository bloat.


8. Model Versioning

Models follow a simple version naming scheme:

training_pipeline_invoice_v1
training_pipeline_oc_v1

Future versions should increment the version number:

training_pipeline_invoice_v2
training_pipeline_oc_v2

This ensures experiments remain reproducible.


9. Reproducibility

The full pipeline can be reproduced by running:

python scripts/training_pipeline.py --target invoice_count
python scripts/training_pipeline.py --target oc_count

The inference wrapper then loads the saved artifacts to produce predictions.

This ensures that:
	•	training is reproducible
	•	predictions use the correct model version
	•	feature schemas remain consistent between training and inference.


Final note:

This document summarizes the modeling system built during Sprint 1 and Sprint 2, including:
	•	dataset construction
	•	model training
	•	hierarchical adjustments
	•	diagnostics
	•	feature importance analysis
	•	inference integration.

The goal is to maintain a clean, reproducible and extensible modeling pipeline for future development.  