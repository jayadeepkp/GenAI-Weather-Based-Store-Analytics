FROM python:3.12

WORKDIR /valvoline

RUN pip install numpy pandas holidays fastapi[standard] pydantic requests scikit-learn lightgbm

COPY notebooks/valvoline_production/valvoline_models_production.pkl .
COPY notebooks/valvoline_production/processed_data.csv .
COPY data_raw/store_info.csv .
COPY demo/api.py .
