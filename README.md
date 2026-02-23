# GenAI Weather-Based Store Analytics – Valvoline Retail

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest streamlit requests pyarrow tqdm joblib
```

Data

Place customer files in data_raw/:
	•	store_info.csv
	•	store_performance_2018to2022.csv
	•	data_dictionary.xlsx

Build (All Stores)

1) Pull Meteostat Weather (Cached)
```bash
source .venv/bin/activate
python scripts/pull_weather_allstores.py
```
Output:
	•	data_processed/weather_allstores.parquet

2) Train + Save Artifacts

Run the training notebook and confirm outputs:
	•	data_processed/model_hgb_allstores.joblib
	•	data_processed/feature_cols_hgb_allstores.json
	•	data_processed/predictions_hgb_allstores.csv
	•	reports/metrics_hgb_allstores.json

3) Build Demo Table (2022)

Run the demo-table cell and confirm output:
	•	data_processed/demo_features_2022.parquet

Run Demo (Localhost Chat UI)
```bash
ollama list
streamlit run demo/app.py
```

Open:
	•	http://localhost:8501

Example prompts:
	•	hi
	•	Predict store 79609 on 2022-01-06

Tests
```bash
pytest -q
```

Outputs
	•	Weather cache: data_processed/weather_allstores.parquet
	•	Model: data_processed/model_hgb_allstores.joblib
	•	Features: data_processed/feature_cols_hgb_allstores.json
	•	Predictions: data_processed/predictions_hgb_allstores.csv
	•	Metrics: reports/metrics_hgb_allstores.json
	•	Demo table: data_processed/demo_features_2022.parquet
