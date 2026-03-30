import pandas as pd
from pathlib import Path
import sys

from config import *
from store_features import add_store_features
from demand_features import add_demand_features
from calendar_features import add_calendar_features
from weather.meteostat_client import get_daily_weather, WeatherConfig
from weather.features import add_weather_features
from lagroll import add_lagroll

# store info data
store_info = pd.read_csv(DATA_RAW / "store_info.csv")
store_info = store_info.sort_values(["store_id"]).reset_index(drop=True)
store_info_cols = list(store_info.columns)

# store performance data
perf = pd.read_csv(DATA_RAW / "store_performance_2018to2022.csv")
perf["invoice_date"] = pd.to_datetime(perf["invoice_date"]).dt.normalize()
perf_cols = ("invoice_count", "oc_count", "fleet_oc_count")

class DatasetBuilder:
    # all of the features listed in reports/metrics_hgb_hierarchical.json
    default_features = [
        "store_id",
        "rain_bucket", "snow_bucket", "heat_bucket", "cold_bucket", "severity",
        "market_id", "store_state", "time_zone_code", "area_id", "marketing_area_id",
        "dow", "month", "day_of_year", "is_weekend",
        "tmin", "tmax", "tavg", "rain_mm", "snow_cm", "wspd", "temp_range",
        "heavy_rain", "heavy_snow", "extreme_heat", "extreme_cold", "freezing",
        "heavy_rain_weekend", "heavy_snow_weekend", "extreme_heat_weekend", "snow_freezing",
        "bay_count", "bay_count_log", "capacity_pressure", "is_closed_day",
        "heavy_rain_capacity", "heavy_snow_capacity",
        "freezing_capacity", "extreme_heat_capacity", "extreme_cold_capacity",
        "invoice_count_lag_1", "invoice_count_lag_7", "invoice_count_lag_14",
        "invoice_count_rollmean_7", "invoice_count_rollmean_28",
        "non_fleet_oc_lag_1", "non_fleet_oc_lag_7", "non_fleet_oc_lag_14",
        "non_fleet_oc_rollmean_7", "non_fleet_oc_rollmean_28",
        "fleet_oc_count_lag_1", "fleet_oc_count_lag_7", "fleet_oc_count_lag_14",
        "fleet_oc_count_rollmean_7", "fleet_oc_count_rollmean_28",
        "rain_mm_lag_1", "rain_mm_rollmean_3", "rain_mm_rollmean_7",
        "snow_cm_lag_1", "snow_cm_rollmean_3", "snow_cm_rollmean_7",
        "tmin_lag_1", "tmin_rollmean_3", "tmin_rollmean_7",
        "tmax_lag_1", "tmax_rollmean_3", "tmax_rollmean_7"
    ]

    def __init__(self, *, cutoff="2022-01-01", features=default_features, n_stores=None, store_selectors={}, verbose=False):
        self.cutoff = cutoff
        self.verbose = verbose
        
        self.select_stores(n_stores, store_selectors)
        
        self.merged = perf[perf.store_id.isin(self.store_ids)].copy()
        self.merged = self.merged.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

        self.features = features
        
    @property
    def train(self):
        return self.merged[self.merged["invoice_date"] < self.cutoff].reset_index(drop=True).copy()

    @property
    def valid(self):
        return self.merged[self.merged["invoice_date"] >= self.cutoff].reset_index(drop=True).copy()

    
    def select_stores(self, n_stores=None, columns={}):
        store_filter = pd.Series([True] * len(store_info.store_id))
        for col_name in columns:
            if col_name not in store_info_cols:
                raise LookupError(f"Column '{col_name}' not in store_info.csv")

            try:
                store_filter &= store_info[col_name].isin(columns[col_name])
            except TypeError:
                store_filter &= store_info[col_name] == columns[col_name]

        store_ids = list(store_info[store_filter].store_id)
        store_ids = store_ids[0:n_stores] if n_stores != None else store_ids 

        self.store_ids = store_ids

        # trim dataframe if method is called after initialization
        if hasattr(self, "merged"):
            self.merged = self.merged[self.merged.store_id.isin(self.store_ids)].reset_index(drop=True)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        self._features = list(features)

        self.print_verbose("Merging store_performance_2018to2022.csv and store_info.csv...")
        m = self.merged[["store_id", "invoice_date", *perf_cols]].copy()
        m = m.merge(store_info, on="store_id", how="left")

        # calendar, store, and demand features
        self.print_verbose("Adding calendar features...")
        m = add_calendar_features(m)
        
        self.print_verbose("Adding store metadata features...")
        m = add_store_features(m)

        self.print_verbose("Adding demand features...")
        m = add_demand_features(m)

        # weather features
        self.print_verbose("Adding base weather features...")
        weather_df = self.get_weather_df()
        m = m.merge(weather_df, on=["store_id", "invoice_date"], how="inner")

        self.print_verbose("Adding engineered weather features...")
        m = add_weather_features(m)

        # other derived features
        self.print_verbose("Adding other engineered features...")
        m["heavy_rain_capacity"] = m["heavy_rain"] * m["capacity_pressure"]
        m["heavy_snow_capacity"] = m["heavy_snow"] * m["capacity_pressure"]
        m["freezing_capacity"] = m["freezing"] * m["capacity_pressure"]
        m["extreme_heat_capacity"] = m["extreme_heat"] * m["capacity_pressure"]
        m["extreme_cold_capacity"] = m["extreme_cold"] * m["capacity_pressure"]

        # lagroll features
        self.print_verbose("Adding lag / rolling mean features...")
        lagroll_features = [f for f in self.features if "lag" in f or "rollmean" in f]
        m = add_lagroll(m, lagroll_features)

        # trim the merged dataset to conform with setter parameter
        self.print_verbose("Trimming dataset to feature selection...")
        default_cols = ["store_id", "invoice_date", *perf_cols, "non_fleet_oc"]
        m = m[default_cols + [f for f in self.features if f not in default_cols]]
        self.merged = m.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

        self.verbose = False

    def get_weather_df(self, cols=WeatherConfig.keep_cols):
        # pull data outputted from scripts/pull_weather_allstores.py if it exists
        allstores_path = Path(DATA_PROCESSED / "weather_allstores.parquet")
        if allstores_path.exists():
            weather_df = pd.read_parquet(allstores_path)
            weather_df = weather_df[["store_id", "invoice_date", *cols]]
            return weather_df

        # otherwise, fetch from Meteostat using ./weather/meteostat_client.py
        weather_df = pd.DataFrame(columns=["store_id", "invoice_date", *cols])
        start = str(self.merged.invoice_date.min())
        end = str(self.merged.invoice_date.max())
        for sid in self.store_ids:
            store = store_info[store_info.store_id == sid].iloc[0]
            lat = store.store_latitude
            lon = store.store_longitude
            store_weather = get_daily_weather(str(sid), lat, lon, start, end, WeatherConfig(keep_cols=weather_cols))
            store_weather["store_id"] = store_weather["store_id"].astype(int)
            weather_df = weather_df.merge(store_weather, how="outer")

        return weather_df

    def print_verbose(self, *msg):
        print(*msg) if self.verbose else None
    
    def to_file(self, filename=None, split=True):
        if filename == None:
            filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".parquet"

        if not split:
            self.merged.to_parquet(DATA_PROCESSED / filename, index=False)
            return
            
        train_name = "train_" + filename
        valid_name = "valid_" + filename

        if filename.endswith(".parquet"):
            self.train.to_parquet(DATA_PROCESSED / train_name, index=False)
            self.valid.to_parquet(DATA_PROCESSED / valid_name, index=False)
        else:
            self.train.to_csv(DATA_PROCESSED / train_name, index=False)
            self.valid.to_csv(DATA_PROCESSED / valid_name, index=False)

    def to_csv(self, filename=None, split=True):
        if filename == None:
            filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".csv"
        elif not filename.endswith(".csv"):
            filename += ".csv"
            
        self.to_file(filename, split)

    def to_parquet(self, filename=None, split=True):
        if filename == None:
            filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".parquet"
        elif not filename.endswith(".parquet"):
            filename += ".parquet"
        
        self.to_file(filename, split)
