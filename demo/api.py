"""
Valvoline Weather Analytics — FastAPI Server
Wraps trained ML models for GenAI chat interface

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /v1/chat/completions → OpenAI-compatible (for OpenWebUI)
    GET  /v1/models           → OpenAI-compatible model list
    POST /predict/impact      → weather impact % forecast
    POST /predict/7days       → 7-day OC forecast
    POST /predict/historical  → historical store weather profile
    POST /predict/chat        → natural language query handler
    GET  /stores              → list all stores
    GET  /health              → health check
"""

import pickle
import re
import numpy as np
import pandas as pd
import holidays
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════
MODEL_PATH = '/Users/jayadeep/GenAI-Weather-Based-Store-Analytics/notebooks/valvoline_production/valvoline_models_production.pkl'
DATA_PATH  = '/Users/jayadeep/GenAI-Weather-Based-Store-Analytics/notebooks/valvoline_production/'
STORE_INFO = '/Users/jayadeep/GenAI-Weather-Based-Store-Analytics/data_raw/store_info.csv'

# ════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════
print('Loading models...')
with open(MODEL_PATH, 'rb') as f:
    models = pickle.load(f)

model_B       = models['model_B_oc_regression']
model_Q05     = models['model_Q05_lower_bound']
model_Q95     = models['model_Q95_upper_bound']
model_FWD     = models['model_FWD']
model_FWD_Q05 = models['model_FWD_Q05']
model_FWD_Q95 = models['model_FWD_Q95']
FEATURES      = models['features']
FWD_FEATURES  = models['forward_features']
CATEGORICALS  = models['categoricals']
label_encoders= models['label_encoders']
print('✅ Models loaded')

# ════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════
print('Loading data...')
df = pd.read_csv(f'{DATA_PATH}processed_data.csv', parse_dates=['invoice_date'])
print(f'✅ Data loaded — {len(df):,} rows, {df["store_id"].nunique()} stores')

# Load store coordinates
store_info_df = pd.read_csv(STORE_INFO)
store_coords  = store_info_df.set_index('store_id')[
    ['store_latitude','store_longitude']
].to_dict('index')
print(f'✅ Store coordinates loaded — {len(store_coords)} stores')

# ════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════

store_dow_baseline_lookup = (
    df.groupby(['store_id','dow'])['store_dow_baseline']
    .first().to_dict()
)
store_typical_oc_lookup = (
    df[df['year']==2022]
    .groupby(['store_id','dow','month'])['oc_count']
    .median().to_dict()
)

def get_store_dow_baseline(store_id, dow):
    key = (store_id, dow)
    if key in store_dow_baseline_lookup:
        return float(store_dow_baseline_lookup[key])
    rows = df[df['store_id']==store_id]['store_dow_baseline']
    return float(rows.mean()) if len(rows) > 0 else 45.0

def get_typical_oc(store_id, dow, month):
    key = (store_id, dow, month)
    if key in store_typical_oc_lookup:
        return float(store_typical_oc_lookup[key])
    return get_store_dow_baseline(store_id, dow)

def classify_weather(tavg, prcp, snow, wspd):
    has_heavy_rain = prcp > 10
    has_rain       = prcp > 0.1
    has_heavy_snow = snow > 150 and tavg <= 2
    has_snow       = snow > 0   and tavg <= 2
    is_freezing    = tavg <= 0
    is_very_cold   = 0  < tavg <= 7
    is_hot         = 27 < tavg <= 35
    has_high_wind  = wspd > 30
    severity = 0
    if is_freezing:    severity += 2
    elif is_very_cold: severity += 1
    if has_heavy_snow: severity += 2
    elif has_snow:     severity += 1
    if has_heavy_rain: severity += 1
    if has_high_wind:  severity += 1
    if severity >= 3:  return 'severe'
    if has_heavy_snow: return 'heavy_snow'
    if has_snow:       return 'any_snow'
    if has_heavy_rain: return 'heavy_rain'
    if has_rain:       return 'light_rain'
    if has_high_wind:  return 'high_wind'
    if is_freezing:    return 'freezing'
    if is_very_cold:   return 'very_cold'
    if is_hot:         return 'hot'
    return 'clear'

def get_weather_forecast(store_id, days=7):
    """
    Fetch real weather forecast for a store using Open-Meteo API.
    Uses store's actual lat/lon coordinates.
    Free — no API key needed.
    Same units as Meteostat: Celsius, mm precipitation.
    """
    if store_id not in store_coords:
        return None

    lat = store_coords[store_id]['store_latitude']
    lon = store_coords[store_id]['store_longitude']

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,"
            f"temperature_2m_mean,precipitation_sum,snowfall_sum,"
            f"windspeed_10m_max"
            f"&timezone=auto"
            f"&forecast_days={days}"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['daily']

        forecast = []
        for i in range(min(days, len(data['time']))):
            forecast.append({
                'date' : data['time'][i],
                'tavg' : float(data['temperature_2m_mean'][i] or 15.0),
                'tmin' : float(data['temperature_2m_min'][i]  or 10.0),
                'tmax' : float(data['temperature_2m_max'][i]  or 20.0),
                'prcp' : float(data['precipitation_sum'][i]   or 0.0),
                'snow' : float((data['snowfall_sum'][i] or 0.0) * 10),
                'wspd' : float(data['windspeed_10m_max'][i]   or 0.0),
            })
        return forecast
    except Exception as e:
        print(f'Weather forecast error for store {store_id}: {e}')
        return None


NETWORK_BASE = {
    'clear'     : ( 0.00, 0.06),
    'very_cold' : (+0.32, 0.09),
    'hot'       : (+0.36, 0.15),
    'light_rain': (-0.95, 0.08),
    'heavy_rain': (-3.05, 0.14),
    'any_snow'  : (-1.75, 0.23),
    'heavy_snow': (-2.14, 0.36),
    'freezing'  : (-1.46, 0.13),
    'high_wind' : (-2.07, 0.68),
    'severe'    : (-2.64, 0.21),
}

WX_LABELS = {
    'clear'     : 'Clear ☀️',
    'very_cold' : 'Very Cold 🧊',
    'hot'       : 'Hot ☀️',
    'light_rain': 'Light Rain 🌦️',
    'heavy_rain': 'Heavy Rain 🌧️',
    'any_snow'  : 'Snow 🌨️',
    'heavy_snow': 'Heavy Snow ❄️',
    'freezing'  : 'Freezing 🥶',
    'high_wind' : 'High Wind 💨',
    'severe'    : 'Severe Weather ⛈️',
}


def predict_day_forward(store_id, forecast_date, weather):
    date  = pd.Timestamp(forecast_date)
    dow   = date.dayofweek
    month = date.month

    store_rows = df[df['store_id']==store_id]
    if len(store_rows) == 0:
        return None
    store_info = store_rows.iloc[0]
    typical_oc = get_typical_oc(store_id, dow, month)

    tavg = float(weather.get('tavg', 15))
    prcp = float(weather.get('prcp', 0))
    snow = float(weather.get('snow', 0))
    wspd = float(weather.get('wspd', 0))
    tmin = float(weather.get('tmin', tavg - 5))
    tmax = float(weather.get('tmax', tavg + 5))

    is_freezing     = int(tavg <= 0)
    is_very_cold    = int(0  < tavg <= 7)
    is_cold         = int(7  < tavg <= 16)
    is_comfortable  = int(16 < tavg <= 27)
    is_hot          = int(27 < tavg <= 35)
    is_extreme_heat = int(tavg > 35)
    has_rain        = int(prcp > 0.1)
    has_heavy_rain  = int(prcp > 10)
    has_snow        = int(snow > 0   and tavg <= 2)
    has_heavy_snow  = int(snow > 150 and tavg <= 2)
    has_high_wind   = int(wspd > 30)

    severity = 0
    if is_freezing or is_extreme_heat: severity += 2
    elif is_very_cold or is_hot:        severity += 1
    if has_heavy_snow:   severity += 2
    elif has_snow:       severity += 1
    if has_heavy_rain:   severity += 1
    if has_high_wind:    severity += 1
    severity = min(severity, 4)

    ts      = date
    month_n = ts.month
    day     = ts.day
    us_hols              = holidays.US(years=[ts.year])
    is_holiday           = int(ts in us_hols)
    is_day_before_holiday= int((ts + pd.Timedelta(days=1)) in us_hols)
    is_day_after_holiday = int((ts - pd.Timedelta(days=1)) in us_hols)
    is_thanksgiving_week = int(month_n == 11 and 25 <= day <= 26)
    is_christmas_week    = int(month_n == 12 and 24 <= day <= 26)
    is_newyear_week      = int(month_n ==  1 and  1 <= day <=  2)
    is_july4_week        = int(month_n ==  7 and  3 <= day <=  5)
    is_laborday_week     = int(month_n ==  9 and  1 <= day <=  2)
    is_memday_week       = int(month_n ==  5 and 27 <= day <= 28)
    is_blackfriday_week  = int(month_n == 11 and 25 <= day <= 26)
    p_abn = 0.7 if (is_holiday or is_thanksgiving_week or
                    is_christmas_week or is_newyear_week) else 0.1

    s_bay      = int(store_info.get('bay_count', 3))
    s_market   = store_info.get('market_id', 0)
    s_area     = store_info.get('area_id', 0)
    s_region   = store_info.get('region_id', 0)
    s_mktarea  = store_info.get('marketing_area_id', 0)
    s_tz       = store_info.get('tz_code', 0)
    s_sun      = int(store_info.get('is_sunday_closed_store', 0))
    s_fleet    = float(store_info.get('store_fleet_dependency', 0.067))
    s_area_oc  = float(store_info.get('area_avg_oc', 45))
    s_mkt_oc   = float(store_info.get('market_avg_oc', 45))
    s_vs_area  = float(store_info.get('store_vs_area_demand', 1.0))
    s_vs_mkt   = float(store_info.get('store_vs_market_demand', 1.0))
    s_rain_sen = float(store_info.get('store_rain_sensitivity', 0.947))
    s_snow_sen = float(store_info.get('store_snow_sensitivity', 0.960))
    s_vol      = float(store_info.get('store_dow_volatility', 5.0))
    s_growth   = float(store_info.get('store_growth_rate', 1.0))

    feat = {
        'dow': dow, 'month': month_n, 'year': date.year,
        'day_of_year': date.dayofyear,
        'week_of_year': date.isocalendar()[1],
        'quarter': date.quarter,
        'is_weekend': int(dow >= 5), 'is_monday': int(dow==0),
        'is_friday': int(dow==4), 'is_saturday': int(dow==5),
        'is_holiday': is_holiday,
        'is_day_before_holiday': is_day_before_holiday,
        'is_day_after_holiday': is_day_after_holiday,
        'is_thanksgiving_week': is_thanksgiving_week,
        'is_christmas_week': is_christmas_week,
        'is_newyear_week': is_newyear_week,
        'is_july4_week': is_july4_week,
        'is_laborday_week': is_laborday_week,
        'is_memday_week': is_memday_week,
        'is_blackfriday_week': is_blackfriday_week,
        'tavg': tavg, 'tmin': tmin, 'tmax': tmax,
        'temp_range': tmax - tmin,
        'prcp': prcp, 'snow': snow, 'wspd': wspd,
        'is_freezing': is_freezing, 'is_very_cold': is_very_cold,
        'is_cold': is_cold, 'is_comfortable': is_comfortable,
        'is_hot': is_hot, 'is_extreme_heat': is_extreme_heat,
        'has_rain': has_rain, 'has_heavy_rain': has_heavy_rain,
        'has_snow': has_snow, 'has_heavy_snow': has_heavy_snow,
        'has_high_wind': has_high_wind, 'severity': severity,
        'bay_count': s_bay, 'market_id': s_market,
        'area_id': s_area, 'region_id': s_region,
        'marketing_area_id': s_mktarea, 'tz_code': s_tz,
        'is_sunday_closed_store': s_sun,
        'store_dow_baseline': typical_oc,
        'area_avg_oc': s_area_oc, 'market_avg_oc': s_mkt_oc,
        'store_vs_area_demand': s_vs_area,
        'store_vs_market_demand': s_vs_mkt,
        'store_fleet_dependency': s_fleet,
        'temp_x_market': tavg * s_market,
        'rain_x_market': prcp * s_market,
        'snow_x_market': snow * s_market,
        'sev_x_market': severity * s_market,
        'temp_x_region': tavg * s_region,
        'snow_x_region': snow * s_region,
        'rain_x_region': prcp * s_region,
        'sev_x_region': severity * s_region,
        'fleet_dep_x_sev': s_fleet * severity,
        'fleet_dep_x_snow': s_fleet * has_heavy_snow,
        'fleet_dep_x_rain': s_fleet * has_heavy_rain,
        'bay_x_severity': s_bay * severity,
        'store_rain_sensitivity': s_rain_sen,
        'store_snow_sensitivity': s_snow_sen,
        'store_dow_volatility': s_vol,
        'store_growth_rate': s_growth,
        'p_abnormal': p_abn,
    }

    for col, le in label_encoders.items():
        if col in feat:
            try:
                feat[col] = int(le.transform([str(int(feat[col]))])[0])
            except:
                feat[col] = 0

    row_fwd = pd.DataFrame([feat]).reindex(columns=FWD_FEATURES, fill_value=0)
    pred  = float(np.maximum(model_FWD.predict(row_fwd)[0],     0))
    lower = float(np.maximum(model_FWD_Q05.predict(row_fwd)[0], 0))
    upper = float(np.maximum(model_FWD_Q95.predict(row_fwd)[0], 0))
    pct   = ((pred - typical_oc) / typical_oc * 100) if typical_oc else 0

    wx_type = classify_weather(tavg, prcp, snow, wspd)
    return {
        'date'         : str(date.date()),
        'day'          : date.strftime('%A'),
        'weather'      : WX_LABELS.get(wx_type, 'Clear ☀️'),
        'wx_type'      : wx_type,
        'typical_oc'   : round(typical_oc),
        'predicted_oc' : round(pred),
        'lower_90'     : round(lower),
        'upper_90'     : round(upper),
        'pct_vs_normal': round(pct, 1),
        'severity'     : severity,
    }


def get_weather_impact(store_id, weather_7days, start_date):
    store_rows = df[df['store_id']==store_id]
    if len(store_rows) == 0:
        return None
    store_info = store_rows.iloc[0]

    rain_sens      = float(store_info.get('store_rain_sensitivity', 0.947))
    snow_sens      = float(store_info.get('store_snow_sensitivity', 0.960))
    store_rain_pct = (rain_sens - 1) * 100
    store_snow_pct = (snow_sens - 1) * 100

    STORE_IMPACT = {
        'clear'     : 0.00,
        'very_cold' : NETWORK_BASE['very_cold'][0],
        'hot'       : NETWORK_BASE['hot'][0],
        'light_rain': store_rain_pct * 0.40,
        'heavy_rain': store_rain_pct,
        'any_snow'  : store_snow_pct * 0.60,
        'heavy_snow': store_snow_pct,
        'freezing'  : NETWORK_BASE['freezing'][0],
        'high_wind' : NETWORK_BASE['high_wind'][0],
        'severe'    : store_rain_pct * 0.80,
    }

    results = []
    for i, wx in enumerate(weather_7days):
        date     = pd.Timestamp(start_date) + pd.Timedelta(days=i)
        tavg     = float(wx.get('tavg', 15))
        prcp     = float(wx.get('prcp', 0))
        snow     = float(wx.get('snow', 0))
        wspd     = float(wx.get('wspd', 0))
        wx_type  = classify_weather(tavg, prcp, snow, wspd)
        pct      = STORE_IMPACT[wx_type]
        normal   = get_typical_oc(store_id, date.dayofweek, date.month)
        expected = round(normal * (1 + pct / 100))
        ci       = normal * 0.15 + 3
        results.append({
            'date'       : str(date.date()),
            'day'        : date.strftime('%A'),
            'weather'    : WX_LABELS.get(wx_type, 'Clear'),
            'wx_type'    : wx_type,
            'normal_oc'  : round(normal),
            'expected_oc': expected,
            'low_oc'     : round(max(0, expected - ci)),
            'high_oc'    : round(expected + ci),
            'pct_impact' : round(pct, 1),
        })
    return results


def get_historical_impact(store_id):
    store_rows = df[df['store_id']==store_id]
    if len(store_rows) == 0:
        return None

    store_data = df[
        (df['store_id']==store_id) &
        (df['is_abnormal_day']==0) &
        (df['oc_count'] > 0)
    ]
    normal_avg = store_data[store_data['severity']==0]['oc_count'].mean()

    conditions = {
        'Normal (no weather)': store_data['severity']==0,
        'Light Rain'         : (store_data['has_rain']==1) & (store_data['has_heavy_rain']==0),
        'Heavy Rain'         : store_data['has_heavy_rain']==1,
        'Any Snow'           : store_data['has_snow']==1,
        'Freezing'           : store_data['is_freezing']==1,
        'Very Cold'          : store_data['is_very_cold']==1,
        'Hot'                : store_data['is_hot']==1,
        'Severe Weather'     : store_data['severity']>=3,
    }

    results = []
    for label, mask in conditions.items():
        subset = store_data[mask]['oc_count']
        if len(subset) < 5:
            continue
        avg = subset.mean()
        pct = (avg - normal_avg) / normal_avg * 100
        results.append({
            'condition'    : label,
            'avg_oc'       : round(avg, 1),
            'pct_vs_normal': round(pct, 1),
            'n_days'       : len(subset),
        })
    return results


def build_system_prompt(store_id):
    """Build rich system prompt with real store data for this store."""
    store_rows = df[df['store_id']==store_id]
    if len(store_rows) == 0:
        return None, None, None

    store     = store_rows.iloc[0]
    city      = store['store_city']
    state     = store['store_state']
    rain_sens = float(store.get('store_rain_sensitivity', 0.947))
    snow_sens = float(store.get('store_snow_sensitivity', 0.960))
    rain_pct  = round((rain_sens - 1) * 100, 1)
    snow_pct  = round((snow_sens - 1) * 100, 1)

    history  = get_historical_impact(store_id)
    hist_str = '\n'.join([
        f"  {h['condition']}: {h['pct_vs_normal']:+.1f}% ({h['n_days']} days)"
        for h in (history or [])
    ])

    dow_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    typical_str = '\n'.join([
        f"  {name}: {round(get_store_dow_baseline(store_id, i))} OC"
        for i, name in enumerate(dow_names)
    ])

    system_prompt = f"""You are a Valvoline Instant Oil Change weather analytics assistant.
You help store managers understand how weather affects their store visits.
You are backed by a machine learning model trained on 5 years of real Valvoline data (2018-2022).
Today's date: {datetime.now().strftime('%A, %B %d, %Y')}

STORE INFORMATION:
- Store ID   : {store_id}
- Location   : {city}, {state}
- Rain impact: {rain_pct}% on heavy rain days (network average: -3.1%)
- Snow impact: {snow_pct}% on snow days (network average: -1.9%)

TYPICAL OC BY DAY OF WEEK (this store's 2022 baseline):
{typical_str}

HISTORICAL WEATHER IMPACT FOR THIS STORE (2018-2022):
{hist_str}

NETWORK-WIDE FINDINGS (579,000 store-days, 95% confidence):
- Heavy Rain   : -3.05% visits (permanent loss — no rebound after rain)
- Heavy Snow   : -2.14% visits (demand shifts forward — rebound after storm)
- Any Snow     : -1.91% visits
- Freezing     : -1.46% visits
- Light Rain   : -0.95% visits
- High Wind    : -2.07% visits
- Day 1 after heavy snow: +3.65%
- Day 2 after heavy snow: +3.34%
- Day 3 after heavy snow: +5.17% (peak rebound)
- Day after rain: NO rebound (+0.02%, not significant)

KEY INSIGHT — FLEET vs RETAIL:
- Rain: fleet customers +4.0%, retail -3.8% (rain is a retail problem)
- Snow: fleet -1.5%, retail -5.7% (snow affects everyone equally)
- High-fleet stores are more weather-resilient in rain

ANSWER INSTRUCTIONS:
- Give specific numbers from THIS store's data above
- Keep answers to 3-5 sentences — be direct and clear
- Say "on average" not "definitely" — individual days vary
- Do NOT use technical terms (MAE, model, confidence interval, p-value)
- Speak like a knowledgeable colleague, not a data scientist
- Always base answers on the real data provided above
- When asked about tomorrow or next week, use the typical OC numbers above"""

    return system_prompt, city, state


# ════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════
app = FastAPI(
    title       = 'Valvoline Weather Analytics API',
    description = 'GenAI-backed weather impact forecasting for store managers',
    version     = '1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)


# ── Request Models ──
class WeatherDay(BaseModel):
    tavg: float
    prcp: float = 0.0
    snow: float = 0.0
    wspd: float = 0.0
    tmin: Optional[float] = None
    tmax: Optional[float] = None

class ImpactRequest(BaseModel):
    store_id  : int
    start_date: str
    weather   : List[WeatherDay]

class ForecastRequest(BaseModel):
    store_id  : int
    start_date: str
    weather   : List[WeatherDay]

class ChatRequest(BaseModel):
    store_id  : int
    message   : str
    weather   : Optional[List[WeatherDay]] = None
    start_date: Optional[str] = None


# ════════════════════════════════════════════════
# OPENAI-COMPATIBLE ENDPOINTS (for OpenWebUI)
# ════════════════════════════════════════════════

@app.get('/v1/models')
def list_models_openai():
    """OpenAI-compatible model list — required by OpenWebUI."""
    return {
        'object': 'list',
        'data': [{
            'id'      : 'valvoline-weather',
            'object'  : 'model',
            'created' : 1700000000,
            'owned_by': 'valvoline',
        }]
    }


@app.post('/v1/chat/completions')
async def openai_chat(request: dict):
    """
    OpenAI-compatible chat completions endpoint.
    Used by OpenWebUI to connect to this API.
    Automatically extracts store_id from conversation.
    Routes through Ollama llama3.1:8b with real store data.
    """
    messages = request.get('messages', [])

    # Get the last user message
    last_message = ''
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            last_message = msg.get('content', '')
            break

    # Extract store_id from message (5-6 digit number)
    store_match = re.search(r'\b(\d{5,6})\b', last_message)
    store_id    = int(store_match.group(1)) if store_match else 79609

    # Verify store exists — fallback to demo store
    if store_id not in df['store_id'].values:
        store_id = 79609

    # Build system prompt with real store data
    system_prompt, city, state = build_system_prompt(store_id)

    # Auto-fetch real 7-day weather forecast for this store's location
    try:
        forecast_data = get_weather_forecast(store_id, days=7)
        if forecast_data:
            start_date   = forecast_data[0]['date']
            impact       = get_weather_impact(store_id, forecast_data, start_date)
            if impact:
                forecast_str = (
                        f'\n\n⚠️ IMPORTANT — YOU MUST USE THIS REAL FORECAST DATA:\n'
                        f'Live 7-day weather forecast for {city}, {state} '
                        f'starting TODAY {datetime.now().strftime("%A %B %d")}:\n'
                    )
                for f in impact:
                    forecast_str += (
                        f"  {f['day']} {f['date']}: {f['weather']} → "
                        f"expect {f['expected_oc']} OC "
                        f"({f['pct_impact']:+.1f}% vs your normal {f['normal_oc']})\n"
                    )
                forecast_str += (
                        f'\nToday is {datetime.now().strftime("%A %B %d %Y")}.\n'
                        f'The forecast above starts TODAY and covers the next 7 days.\n'
                        f'When manager asks about "next week" or "upcoming days", '
                        f'use ONLY these real forecast numbers above.\n'
                        f'Do NOT shift dates — use the exact dates shown above.\n'
                    )
                system_prompt += forecast_str
                print(f'✅ Auto-fetched forecast for store {store_id} ({city}, {state})')
    except Exception as e:
        print(f'Auto-forecast warning: {e}')

    # Build full message list for Ollama
    ollama_messages = [{'role': 'system', 'content': system_prompt}]
    for msg in messages:
        if msg.get('role') in ['user', 'assistant']:
            ollama_messages.append({
                'role'   : msg['role'],
                'content': msg['content']
            })

    # Call Ollama
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model'   : 'llama3.1:8b',
                'messages': ollama_messages,
                'stream'  : False,
                'options' : {
                    'temperature': 0.3,
                    'num_predict': 400,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()['message']['content']
    except requests.exceptions.ConnectionError:
        answer = 'Ollama is not running. Please start with: ollama serve'
    except Exception as e:
        answer = f'Error connecting to Ollama: {str(e)}'

    # Return OpenAI-compatible format
    return {
        'id'     : 'chatcmpl-valvoline',
        'object' : 'chat.completion',
        'created': int(datetime.now().timestamp()),
        'model'  : 'valvoline-weather',
        'choices': [{
            'index'        : 0,
            'message'      : {
                'role'   : 'assistant',
                'content': answer
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens'    : 0,
            'completion_tokens': 0,
            'total_tokens'     : 0
        }
    }


# ════════════════════════════════════════════════
# STANDARD ENDPOINTS
# ════════════════════════════════════════════════

@app.get('/health')
def health():
    return {
        'status' : 'ok',
        'models' : 'loaded',
        'stores' : df['store_id'].nunique(),
        'version': '1.0.0'
    }


@app.get('/stores')
def list_stores():
    stores = df.groupby('store_id').agg(
        city  = ('store_city',  'first'),
        state = ('store_state', 'first'),
    ).reset_index()
    return {'stores': stores.to_dict('records')}


@app.get('/stores/{store_id}')
def get_store(store_id: int):
    rows = df[df['store_id']==store_id]
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail=f'Store {store_id} not found')
    store     = rows.iloc[0]
    rain_sens = float(store.get('store_rain_sensitivity', 0.947))
    snow_sens = float(store.get('store_snow_sensitivity', 0.960))
    dow_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    typical_by_dow = {
        name: round(get_store_dow_baseline(store_id, i))
        for i, name in enumerate(dow_names)
    }
    return {
        'store_id'         : store_id,
        'city'             : store['store_city'],
        'state'            : store['store_state'],
        'bay_count'        : int(store.get('bay_count', 3)),
        'rain_impact_pct'  : round((rain_sens - 1) * 100, 1),
        'snow_impact_pct'  : round((snow_sens - 1) * 100, 1),
        'typical_oc_by_dow': typical_by_dow,
    }


@app.post('/predict/impact')
def predict_impact(req: ImpactRequest):
    weather_list = [w.dict() for w in req.weather]
    results      = get_weather_impact(req.store_id, weather_list, req.start_date)
    if results is None:
        raise HTTPException(status_code=404, detail=f'Store {req.store_id} not found')
    store = df[df['store_id']==req.store_id].iloc[0]
    return {
        'store_id'  : req.store_id,
        'city'      : store['store_city'],
        'state'     : store['store_state'],
        'start_date': req.start_date,
        'forecast'  : results,
    }


@app.post('/predict/7days')
def predict_7days(req: ForecastRequest):
    if len(req.weather) != 7:
        raise HTTPException(status_code=400, detail='Exactly 7 weather days required')
    results = []
    for i, wx in enumerate(req.weather):
        date   = pd.Timestamp(req.start_date) + pd.Timedelta(days=i)
        result = predict_day_forward(req.store_id, date, wx.dict())
        if result:
            results.append(result)
    store = df[df['store_id']==req.store_id].iloc[0]
    return {
        'store_id'  : req.store_id,
        'city'      : store['store_city'],
        'state'     : store['store_state'],
        'start_date': req.start_date,
        'forecast'  : results,
    }


@app.post('/predict/historical')
def predict_historical(store_id: int):
    results = get_historical_impact(store_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f'Store {store_id} not found')
    store     = df[df['store_id']==store_id].iloc[0]
    rain_sens = float(store.get('store_rain_sensitivity', 0.947))
    snow_sens = float(store.get('store_snow_sensitivity', 0.960))
    return {
        'store_id'        : store_id,
        'city'            : store['store_city'],
        'state'           : store['store_state'],
        'rain_impact_pct' : round((rain_sens - 1) * 100, 1),
        'snow_impact_pct' : round((snow_sens - 1) * 100, 1),
        'network_rain_avg': -0.9,
        'network_snow_avg': -1.8,
        'history'         : results,
    }


@app.post('/predict/chat')
def chat(req: ChatRequest):
    system_prompt, city, state = build_system_prompt(req.store_id)
    if system_prompt is None:
        raise HTTPException(status_code=404, detail=f'Store {req.store_id} not found')

    # Auto-fetch real forecast even when no weather provided
    try:
        forecast_data = get_weather_forecast(req.store_id, days=7)
        if forecast_data:
            start_date = forecast_data[0]['date']
            impact     = get_weather_impact(req.store_id, forecast_data, start_date)
            if impact:
                forecast_str = (
                    f'\n\n⚠️ IMPORTANT — USE THIS REAL FORECAST DATA:\n'
                    f'Live 7-day weather forecast for {city}, {state} '
                    f'starting TODAY {datetime.now().strftime("%A %B %d")}:\n'
                )
                for f in impact:
                    forecast_str += (
                        f"  {f['day']} {f['date']}: {f['weather']} → "
                        f"expect {f['expected_oc']} OC "
                        f"({f['pct_impact']:+.1f}% vs your normal {f['normal_oc']})\n"
                    )
                forecast_str += (
                    f'\nToday is {datetime.now().strftime("%A %B %d %Y")}.\n'
                    f'Use ONLY these forecast numbers. Do NOT ignore weather impact.\n'
                )
                system_prompt += forecast_str
    except Exception as e:
        print(f'Auto-forecast warning: {e}')

    if req.weather and req.start_date:
        weather_list = [w.dict() for w in req.weather]
        forecast     = get_weather_impact(req.store_id, weather_list, req.start_date)
        if forecast:
            forecast_str = '\nWEATHER FORECAST:\n' + '\n'.join([
                f"  {f['day']} {f['date']}: {f['weather']} → "
                f"expect {f['expected_oc']} OC ({f['pct_impact']:+.1f}% vs normal {f['normal_oc']})"
                for f in forecast
            ])
            system_prompt += forecast_str

    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model'   : 'llama3.1:8b',
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': req.message}
                ],
                'stream'  : False,
                'options' : {'temperature': 0.3, 'num_predict': 400}
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()['message']['content']
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail='Ollama not running')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        'store_id': req.store_id,
        'city'    : city,
        'state'   : state,
        'question': req.message,
        'answer'  : answer,
    }


# ════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
