from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from meteostat import Point

# Meteostat API differs by version: Daily may live in meteostat or meteostat.daily
try:
    from meteostat import Daily  # newer versions
except Exception:
    from meteostat.daily import Daily  # older versions


@dataclass(frozen=True)
class WeatherConfig:
    cache_dir: Path = Path("data_raw/weather_cache")
    keep_cols: Tuple[str, ...] = ("tavg", "tmin", "tmax", "prcp", "snow", "wspd")


def _norm_date(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()


def _full_dates(start_date: str, end_date: str) -> pd.DataFrame:
    s = _norm_date(start_date)
    e = _norm_date(end_date)
    return pd.DataFrame({"invoice_date": pd.date_range(s, e, freq="D")})


def _cache_path(cfg: WeatherConfig, store_id: str) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"store_id={store_id}_daily.csv"


def _fetch_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    p = Point(lat, lon)
    s = _norm_date(start_date)
    e = _norm_date(end_date)

    df = Daily(p, s, e).fetch()
    if df is None or len(df) == 0:
        return pd.DataFrame({"invoice_date": []})

    df = df.reset_index().rename(columns={"time": "invoice_date"})
    df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.normalize()
    return df


def get_daily_weather(
    store_id: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cfg: Optional[WeatherConfig] = None,
    refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per day from start_date..end_date.
    Uses local cache unless refresh=True.
    """

    cfg = cfg or WeatherConfig()
    store_id = str(store_id)

    cpath = _cache_path(cfg, store_id)

    if cpath.exists() and not refresh:
        if verbose:
            print(f"[CACHE_HIT] store_id={store_id}")
        df = pd.read_csv(cpath, parse_dates=["invoice_date"])
        df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.normalize()
    else:
        if verbose:
            print(f"[FETCH] store_id={store_id} lat={lat} lon={lon}")
        df = _fetch_daily(lat, lon, start_date, end_date)

        keep = ["invoice_date"] + [c for c in cfg.keep_cols if c in df.columns]
        df = df[[c for c in keep if c in df.columns]].copy()

        df.to_csv(cpath, index=False)

    # enforce continuous daily dates
    base = _full_dates(start_date, end_date)
    out = base.merge(df, on="invoice_date", how="left")

    out.insert(0, "store_id", store_id)

    if verbose:
        print(f"[COVERAGE] store_id={store_id}, rows={len(out)}")

    return out