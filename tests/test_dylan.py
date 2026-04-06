"""
Test Case — Dylan Devereaux
Title: Model Inference API Delivers Complete Prediction Contract for Any Store and Date
Severity: Critical

Instructions:
1. Start the API: cd demo && uvicorn api:app --host 0.0.0.0 --port 8000 --reload
2. Wait for "Application startup complete" in the terminal
3. Run the test from the project root directory: pytest tests/test_dylan.py
4. Verify all required fields are present and error handling works
"""

import requests

BASE = 'http://localhost:8000'

# test that API has loaded everything correctly
def test_api_health():
    # /health
    # API must be healthy with all models loaded
    health = requests.get(f'{BASE}/health', timeout=5)
    assert health.status_code == 200
    assert health.json()['status']  == 'ok'
    assert health.json()['models']  == 'loaded'
    assert health.json()['stores']  >= 400, \
        'Fewer than 400 stores loaded — data issue'

    # /v1/models
    # OpenAI model list works for OpenWebUI
    models = requests.get(f'{BASE}/v1/models', timeout=5)
    assert models.status_code == 200
    assert models.json()['data'][0]['id'] == 'valvoline-weather'

    # /stores
    # all stores (or at least most) are loaded correctly
    stores = requests.get(f'{BASE}/stores')
    assert stores.status_code == 200

    store_list = stores.json()['stores']
    assert len(stores.json()['stores']) >= 400, \
        'Fewer than 400 stores loaded — data issue'

    for store in store_list:
        assert type(store["store_id"]) == int
        assert type(store["city"]) == str and store["city"] != ""
        assert type(store["state"]) == str and store["state"] != ""

# test that API can retrieve store data and historical profile
def test_api_store_profile():
    sid = 79609

    # /stores/{store_id}
    # Store info returns correct details
    store = requests.get(f'{BASE}/stores/{sid}', timeout=5)
    assert store.status_code == 200
    s = store.json()
    assert s['city']           == 'Lexington'
    assert s['state']          == 'KY'
    assert s['rain_impact_pct'] < 0, \
        'Rain impact must be negative for this store'
    assert s['snow_impact_pct'] < 0, \
        'Snow impact must be negative for this store'

    # /predict/historical
    # store historical profile returns correct details
    store_hist = store = requests.post(f'{BASE}/predict/historical?store_id={sid}', timeout=5)
    assert store.status_code == 200
    s = store.json()
    assert s['city']           == 'Lexington'
    assert s['state']          == 'KY'
    assert s['rain_impact_pct'] < 0, \
        'Rain impact must be negative for this store'
    assert s['snow_impact_pct'] < 0, \
        'Snow impact must be negative for this store'
    assert s['network_rain_avg'] <= 0, \
        'Rain impact average must be negative'
    assert s['network_snow_avg'] <= 0, \
        'Snow impact average must be negative'

    for entry in s['history']:
        assert entry['condition'] in ['Normal (no weather)', 'Light Rain', 'Heavy Rain',
                                      'Any Snow', 'Freezing', 'Very Cold', 'Hot', 'Severe Weather']
        assert type(entry['avg_oc']) in [float, int]
        assert entry['pct_vs_normal'] <= 0
        assert type(entry['n_days']) == int

# test that api forecast conforms to contract
def test_api_delivers_complete_prediction_contract():
    # Week prediction returns full contract
    r = requests.get(
        f'{BASE}/predict/week/79609/2026-04-07',
        timeout=15
    )
    assert r.status_code == 200
    data = r.json()
 
    # All top-level contract fields present
    for field in ['store_id', 'city', 'state', 'week_start',
                  'week_end', 'predictions', 'weekly_total',
                  'model_mae', 'note']:
        assert field in data, f'Missing contract field: {field}'
 
    # Model MAE within acceptable range — not hardcoded
    assert 4.0 <= data['model_mae'] <= 7.0, \
        f'MAE {data["model_mae"]} outside acceptable range'
 
    # Every prediction day has full contract fields
    for p in data['predictions']:
        for field in ['date', 'day', 'weather', 'temp_c',
                      'precip_mm', 'wind_kmh', 'normal_oc',
                      'pct_impact', 'predicted_oc',
                      'range_low', 'range_high']:
            assert field in p, \
                f'Missing prediction field: {field}'
 
    # Invalid store returns error — no server crash
    r_bad = requests.get(
        f'{BASE}/predict/week/99999/2026-04-07',
        timeout=5
    )
    assert r_bad.status_code in [404, 500], \
        'Invalid store should return error not 200'
