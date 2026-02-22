import os, ssl, certifi, urllib.request
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from meteostat import Point
from meteostat.interface.daily import Daily

# ---- SSL hard fix ----
cafile = certifi.where()
os.environ["SSL_CERT_FILE"] = cafile
os.environ["REQUESTS_CA_BUNDLE"] = cafile
ctx = ssl.create_default_context(cafile=cafile)
urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)))

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data_raw"
DATA_PROCESSED = ROOT / "data_processed"
CACHE_DIR = DATA_PROCESSED / "weather_cache_daily"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

store_info = pd.read_csv(DATA_RAW / "store_info.csv")
perf = pd.read_csv(DATA_RAW / "store_performance_2018to2022.csv")

store_info["store_id"] = store_info["store_id"].astype(int)
perf["store_id"] = perf["store_id"].astype(int)
perf["invoice_date"] = pd.to_datetime(perf["invoice_date"]).dt.normalize()

stores_all = (
    store_info.dropna(subset=["store_latitude","store_longitude"])
    .merge(perf[["store_id"]].drop_duplicates(), on="store_id", how="inner")
    .drop_duplicates("store_id")
    .sort_values("store_id")
)

start_dt = perf["invoice_date"].min().to_pydatetime()
end_dt   = perf["invoice_date"].max().to_pydatetime()

def cache_path(store_id: int) -> Path:
    return CACHE_DIR / f"store_{store_id}.parquet"

frames = []
success = 0
failed = 0
empty = 0
cache_hits = 0

for _, r in tqdm(stores_all.iterrows(), total=len(stores_all), desc="Meteostat all-stores (script)"):
    sid = int(r["store_id"])
    lat = float(r["store_latitude"])
    lon = float(r["store_longitude"])
    cp = cache_path(sid)

    if cp.exists():
        try:
            w = pd.read_parquet(cp)
            if len(w) > 0:
                frames.append(w); success += 1; cache_hits += 1
            else:
                empty += 1
        except Exception:
            failed += 1
        continue

    try:
        w = Daily(Point(lat, lon), start_dt, end_dt).fetch()
        if w is None or len(w) == 0:
            empty += 1
            continue

        w = w.reset_index().rename(columns={"time":"invoice_date"})
        w["invoice_date"] = pd.to_datetime(w["invoice_date"]).dt.normalize()
        w["store_id"] = sid

        keep = ["store_id","invoice_date","tavg","tmin","tmax","prcp","wspd","snow"]
        w = w[[c for c in keep if c in w.columns]].drop_duplicates(["store_id","invoice_date"])

        w.to_parquet(cp, index=False)
        frames.append(w); success += 1

    except Exception:
        failed += 1
        continue

print("SUCCESS stores:", success)
print("CACHE hits:", cache_hits)
print("EMPTY stores:", empty)
print("FAILED stores:", failed)

if len(frames) == 0:
    raise SystemExit("No weather frames collected. Still blocked by SSL/network.")

weather_all = pd.concat(frames, ignore_index=True).drop_duplicates(["store_id","invoice_date"])
out = DATA_PROCESSED / "weather_allstores.parquet"
weather_all.to_parquet(out, index=False)
print("Saved:", out)