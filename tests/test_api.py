import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.main import app

client = TestClient(app)

def test_api_cache_hit():
    badreq = {
        "store_id": 98317,
        "date": "2020-07-15",
    }

    response = client.post("/predict", json=badreq)
    assert response.status_code == 400, print(response.json())

    goodreq = {
        "store_id": 98317,
        "date": "2022-07-15",
    }

    response = client.post("/predict", json=goodreq)
    assert response.status_code == 200, print(response.json())

    body = response.json()
    assert type(body["pred_invoice"]) == float and body["pred_invoice"] != float('nan')
    assert type(body["pred_oc"]) == float and body["pred_oc"] != float('nan')
    assert type(body["oc_lo95"]) == float and body["oc_lo95"] != float('nan')
    assert type(body["oc_hi95"]) == float and body["oc_hi95"] != float('nan')
    assert type(body["severity"]) == str and body["severity"] != ""
    assert type(body["oc_pchg"]) == float and body["oc_pchg"] != float('nan')
    assert type(body["actual_invoice"]) == float and body["actual_invoice"] != float('nan')
    assert type(body["actual_oc"]) == float and body["actual_oc"] != float('nan')


def test_api_forecast():
    badreq = {
        "store_id": 98317,
        "date": "2024-01-22"
    }

    response = client.post("/predict", json=badreq)
    assert response.status_code == 400, print(response.json())

    goodreq = badreq | {
        "weather": {
            "tmin": -8,
            "tmax": 8,
            "tavg": 0,
            "prcp": 0,
            "wspd": 21,
            "snow": 1
        }
    }

    response = client.post("/predict", json=goodreq)
    assert response.status_code == 200, print(response.json())

    body = response.json()
    assert type(body["pred_invoice"]) == float and body["pred_invoice"] != float('nan')
    assert type(body["pred_oc"]) == float and body["pred_oc"] != float('nan')
    assert type(body["oc_lo95"]) == float and body["oc_lo95"] != float('nan')
    assert type(body["oc_hi95"]) == float and body["oc_hi95"] != float('nan')
    assert type(body["severity"]) == str and body["severity"] != ""
    assert type(body["oc_pchg"]) == float and body["oc_pchg"] != float('nan')
