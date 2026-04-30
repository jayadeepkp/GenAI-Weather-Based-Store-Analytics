# GenAI Weather-Based Store Analytics
### Valvoline Instant Oil Change · CS499 Senior Design · University of Kentucky · April 2026

> **Does weather significantly impact oil change visits?**
> YES — proven at 95% statistical confidence across 579,000 store-days.

A machine learning pipeline and GenAI chat interface that predicts daily oil change visits for 439 Valvoline stores across 19 states using live weather forecasts.

---

## Team

| Name | Role |
|------|------|
| Jayadeep Kothapalli | ML Pipeline & Data Analysis |
| Dylan Devereaux | GenAI Interface & API |
| Rudwika Manne | Data Processing & Testing |
| Harshini Ponnam | Evaluation & Validation |

**Client:** Valvoline Retail Services — Dom Trivison, Colton Hinrichs

---

## Key Results

| Metric | Value |
|--------|-------|
| R² (2023 real unseen data) | **0.830** — beats UK benchmark 0.822 |
| MAE (batch model) | **5.08 visits** on 2022 holdout |
| MAE (2023 validation) | **5.29 visits** on 155,230 real store-days |
| 90% CI Coverage | **84.7%** (90% with scaling) |
| Stores validated | **439 stores · 19 states** |
| Automated tests | **27/27 passing** |

---

## What We Built

```
1. LightGBM ML pipeline trained on 5 years of real Valvoline data (2018-2022)
2. 6 production models: batch, forward, and 4 quantile models
3. FastAPI server with 7 endpoints
4. GenAI chat interface via OpenWebUI + Ollama
5. Live weather integration via Open-Meteo API (free, no key needed)
6. Store-specific sensitivity profiles for all 439 stores
```

---

## Project Structure

```
GenAI-Weather-Based-Store-Analytics/
│
├── demo/
│   └── api.py                          ← FastAPI server (run this)
│
├── notebooks/
│   ├── valvoline_production/
│   │   ├── valvoline_production.ipynb  ← Main training notebook
│   │   ├── valvoline_models_production.pkl  ← 6 trained models
│   │   └── processed_data.csv          ← Training data
│   │
│   └── valvoline_evaluation/
│       ├── valvoline_evaluation.ipynb  ← Evaluation notebook
│       └── predictions_2022.csv        ← 2022 holdout predictions
│
├── data_raw/
│   └── store_info.csv                  ← Store GPS coordinates
│
├── tests/
│   └── test_api.py                     ← 27 automated tests
│
└── README.md
```

---

## Prerequisites

### System Requirements

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 or 3.12 | FastAPI server |
| Ollama | Latest | Local LLM |
| Docker | Latest | OpenWebUI |

> ⚠️ **Python 3.13 is NOT supported** — OpenWebUI requires Python >=3.11, <3.13

---

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/jayadeepkp/GenAI-Weather-Based-Store-Analytics.git
cd GenAI-Weather-Based-Store-Analytics
```

### Step 2 — Create Virtual Environment (Python 3.12)

```bash
# Install Python 3.12 if needed (Mac)
brew install python@3.12

# Create venv
python3.12 -m venv .venv

# Activate venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### Step 3 — Install Python Dependencies

```bash
pip install fastapi uvicorn pydantic requests numpy pandas \
            lightgbm scikit-learn holidays python-multipart
```

### Step 4 — Install Ollama

```bash
# Mac
brew install ollama

# Or download from https://ollama.ai

# Pull the model
ollama pull llama3.1:8b
```

### Step 5 — Install OpenWebUI via Docker

```bash
docker run -d \
  -p 3000:8080 \
  -e WEBUI_AUTH=False \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

---

## Running the Demo

Open **3 separate terminals** and run in order:

### Terminal 1 — Start Ollama LLM

```bash
ollama serve
```

### Terminal 2 — Start FastAPI Server

```bash
source /path/to/GenAI-Weather-Based-Store-Analytics/.venv/bin/activate

cd /path/to/GenAI-Weather-Based-Store-Analytics/demo

uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

### Terminal 3 — Start OpenWebUI

```bash
# If already installed
docker start open-webui

# First time only (run the docker run command from Installation Step 5)
```

### Open the Chat Interface

```
http://localhost:3000
```

### Verify Everything is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "models": "loaded", "stores": 439, "version": "1.0.0"}
```

---

## Demo Questions

Try these in the chat interface:

```
1. "What should I expect this week at store 79609?"
   → Shows 7-day forecast with confidence ranges

2. "How many oil changes should I expect tomorrow at
    store 79609 and how does the weather affect it?"
   → Shows next-day prediction with weather context

3. "It is forecasted to rain heavily this Saturday at
    store 79609. How many oil changes should I expect
    compared to a normal Saturday?"
   → Shows store-specific rain impact
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — models loaded, store count |
| GET | `/v1/models` | OpenAI-compatible model list |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (used by OpenWebUI) |
| GET | `/stores` | List all 439 stores |
| GET | `/stores/{store_id}` | Store details + sensitivity profile |
| POST | `/predict/impact` | Weather impact % forecast |
| POST | `/predict/7days` | 7-day OC forecast |
| POST | `/predict/historical` | Historical store weather profile |
| POST | `/predict/chat` | Natural language query handler |
| GET | `/predict/week/{store_id}/{start_date}` | Weekly OC prediction |

### Example API Call

```bash
# Get store info
curl http://localhost:8000/stores/79609 | python3 -m json.tool

# Get weekly forecast
curl http://localhost:8000/predict/week/79609/2026-04-29 | python3 -m json.tool

# Chat endpoint
curl -s -X POST http://localhost:8000/predict/chat \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 79609,
    "message": "Should I expect heavy rain to affect visits this week?"
  }' | python3 -m json.tool
```

---

## The Models

| Model | Purpose | Features | MAE |
|-------|---------|----------|-----|
| Model B (Batch) | Next-day prediction | 100 (uses OC lags) | 5.08 |
| Model Q05 | Lower confidence bound | 100 | — |
| Model Q95 | Upper confidence bound | 100 | — |
| Model FWD | 7-day blind forecast | 71 (no lags) | 5.73 |
| Model FWD_Q05 | Forward lower bound | 71 | — |
| Model FWD_Q95 | Forward upper bound | 71 | — |

All models saved in: `notebooks/valvoline_production/valvoline_models_production.pkl`

---

## Key Weather Findings

| Condition | Network Effect | Significant? |
|-----------|---------------|-------------|
| Heavy Rain | -3.06% | YES |
| High Wind | -2.34% | YES |
| Heavy Snow | -2.18% | YES |
| Any Snow | -1.83% | YES |
| Freezing (≤0°C) | -1.32% | YES |
| Light Rain | -0.99% | YES |
| Day After Heavy Snow | +3.30% | YES |
| Day 2 After Heavy Snow | +3.75% | YES |
| Day 3 After Heavy Snow | +5.65% | YES |
| Day After Heavy Rain | -0.06% | NO — no rebound |
| Before Extreme Rain (1 day) | +2.57% | YES |
| Before Extreme Rain (3 days) | +2.34% | YES |
| Before Heavy Snow (3 days) | +0.56% | YES |
| Before Heavy Snow (1 day) | -2.55% | YES |

### Key Insight

**Rain shifts demand backward** — customers get their oil change before rain arrives.
**Snow shifts demand forward** — customers rush in before a snowstorm and rebound strongly after.

---

## Store Sensitivity

| Metric | Rain | Snow |
|--------|------|------|
| Network average | -4.5% | -10.4% |
| Most sensitive | -23.9% (Eagle, ID) | -49.7% (Richmond, VA) |
| Least sensitive | +15.9% (Burton, MI) | +4.9% (East Lansing, MI) |
| Stores losing >5% | 231 of 430 | — |
| Stores losing >10% | 28 of 430 | 97 of 213 |

**Fleet vs Retail in Rain:**
- Fleet customers: +4.0% in rain
- Retail customers: -3.8% in rain
- Rain = retail problem only

---

## Running Tests

```bash
cd /path/to/GenAI-Weather-Based-Store-Analytics

# Make sure FastAPI is running first (Terminal 2 above)

# Run all 27 tests
python -m pytest tests/test_api.py -v

# Expected: 27 passed
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Models | LightGBM (gradient boosting) |
| API Server | FastAPI + Uvicorn |
| Chat Interface | OpenWebUI |
| LLM | Ollama llama3.1:8b |
| Live Weather | Open-Meteo API (free, no key) |
| Historical Weather | Meteostat API |
| Deployment | Docker |
| Language | Python 3.12 |

---

## Troubleshooting

### FastAPI won't start

```bash
# Make sure you are in the demo/ folder
cd /path/to/GenAI-Weather-Based-Store-Analytics/demo
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# NOT from the root folder — that picks up api/__init__.py instead
```

### OpenWebUI shows no model

```
1. Go to http://localhost:3000
2. Click profile icon → Settings → Connections
3. Add OpenAI connection:
   URL: http://host.docker.internal:8000/v1
   Key: dummy
4. Save → refresh
```

### Ollama not responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Check model is downloaded
ollama list
# Should show: llama3.1:8b
```

### Docker container not starting

```bash
# Check container status
docker ps -a

# Start existing container
docker start open-webui

# View logs
docker logs open-webui --tail 50
```

### Python version error (open-webui install)

```bash
# open-webui requires Python >=3.11, <3.13
# If you have Python 3.13, use Docker instead (see Installation Step 5)
python3 --version
```

---

## Data

| Dataset | Size | Period |
|---------|------|--------|
| Training data | 773,266 rows | 2018-2022 |
| Stores | 439 | 19 states |
| 2022 holdout | 152,898 store-days | 2022 |
| 2023 validation | 155,230 store-days | 2023 |
| Weather source (training) | Meteostat API | 2018-2022 |
| Weather source (live) | Open-Meteo API | Today + 7 days |

---

## What Would Improve Accuracy Further

| Addition | Estimated MAE Improvement |
|----------|--------------------------|
| Valvoline promotional calendar | -2 to -3 visits |
| Daily rolling update | -2 to -3 visits |
| Local events data | -1 to -2 visits |
| Retrain on 2022 data | -0.3 to -0.5 visits |

---

## License

University of Kentucky CS499 Senior Design Project — April 2026.
Built for Valvoline Retail Services.

---

*Dylan Devereaux · Jayadeep Kothapalli · Rudwika Manne · Harshini Ponnam*
*University of Kentucky · Department of Computer Science*
