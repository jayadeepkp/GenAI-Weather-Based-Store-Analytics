from datetime import datetime
import pandas as pd
from .config import *
from .demand_features import add_demand_features
from .weather.meteostat_client import get_daily_weather, WeatherConfig
from .weather.features import add_weather_features

store_info = pd.read_csv(DATA_RAW / "store_info.csv")
store_info_cols = list(store_info.columns)

perf = pd.read_csv(DATA_RAW / "store_performance_2018to2022.csv")
perf["invoice_date"] = pd.to_datetime(perf["invoice_date"]).dt.normalize()
perf_cols = list(perf.columns)

class DatasetBuilder:
    # many, but not all of the features that have been used for training prototypes
    # some are not fully implemented; more on that in the comments for the features setter
    location_features = list(set(store_info_cols) - {"invoice_date", "store_id"})
    calendar_features = ["year", "month", "day_of_year", "dow", "is_weekend"]
    demand_features = ["inv_lag_1", "inv_lag_7", "inv_roll7_mean", "inv_roll14_mean", "inv_roll28_mean"]
    weather_features = list(WeatherConfig.keep_cols)
    default_features = calendar_features + demand_features + weather_features
    
    def __init__(self, *, cutoff="2022-01-01", features=default_features, n_stores=None, store_selectors={}):
        self.cutoff = cutoff

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
        store_filter = store_info.store_id == store_info.store_id
        store_ids = []

        for col_name in columns:
            if col_name not in store_info_cols:
                raise LookupError(f"Column '{col_name}' not in store_info.csv")
            elif not isinstance(columns[col_name], (str, int, float, list, tuple, set)):
                raise TypeError(f"Invalid type for argument '{col_name}={columns[col_name]}'")

            try:
                store_filter &= store_info[col_name].isin(columns[col_name])
            except TypeError:
                store_filter &= store_info[col_name] == columns[col_name]

        store_ids = list(store_info[store_filter].store_id)
        store_ids = store_ids[0:n_stores] if n_stores != None else store_ids 

        self.store_ids = store_ids

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        self._features = list(features)

        # location features
        location_cols = list(set(features).intersection(self.location_features))
        self.merged = self.merged.merge(store_info[["store_id", *location_cols]], how="left")

        # demand features
        # due to how add_demand_features works, will insert all columns regardless of selection
        has_demand_features = any(map(lambda f: f in self.demand_features, features))
        if has_demand_features:
            self.merged = add_demand_features(self.merged)

        # weather features
        # currently uses ./weather/meteostat_client.py functionality
        # but could be modified to function with data produced by scripts/pull_weather_allstores.py
        weather_cols = tuple(set(features).intersection(self.weather_features))
        weather_df = pd.DataFrame(columns=["store_id", "invoice_date", *weather_cols])
        for sid in self.store_ids:
            store = store_info[store_info.store_id == sid].iloc[0]
            lat = store.store_latitude
            lon = store.store_longitude
            store_weather = get_daily_weather(str(sid), lat, lon,
                                              str(self.merged.invoice_date.min()), str(self.merged.invoice_date.max()),
                                              WeatherConfig(keep_cols=weather_cols))
            store_weather["store_id"] = store_weather["store_id"].astype(int)
            weather_df = weather_df.merge(store_weather, how="outer")

        self.merged = self.merged.merge(weather_df, on=["store_id", "invoice_date"], how="left")
        self.merged = add_weather_features(self.merged) # includes dow, month, and is_weekend

    def to_csv(self, filename=None):
        if filename == None:
            filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".csv"

        train_name = "train_" + filename
        valid_name = "valid_" + filename

        self.train.to_csv(DATA_PROCESSED / train_name)
        self.valid.to_csv(DATA_PROCESSED / valid_name)
