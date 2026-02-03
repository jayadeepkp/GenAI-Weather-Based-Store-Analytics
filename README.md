# GenAI Weather-Based Store Analytics – Valvoline Retail

## Project Goal
Use historical store performance + weather data to:
1) quantify weather impact on store activity
2) predict future activity using forecast inputs (same store locations)

## Quick Start (Codespaces)
1. Open repo in GitHub Codespaces
2. Wait for environment setup
3. Use kernel: **Python (capstone)**

## Quick Start (Local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name capstone --display-name "Python (capstone)"
jupyter notebook



z---

## G) Add a “data_raw” warning README

```bash
cat > data_raw/README.md << 'EOF'
## data_raw (DO NOT COMMIT)

Place customer-provided files here locally:
- store_info.csv
- store_performance_2018to2022.csv
- data_dictionary.csv (or xlsx)

This folder is gitignored to protect NDA data.
