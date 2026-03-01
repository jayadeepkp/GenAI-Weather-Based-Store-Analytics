import pandas as pd
from pathlib import Path

from .config import *
from .demand_features import add_demand_features
from .weather.meteostat_client import get_daily_weather, WeatherConfig
from .weather.features import add_weather_features

store_info = pd.read_csv(DATA_RAW / "store_info.csv")
store_info_cols = list(store_info.columns)

perf = pd.read_csv(DATA_RAW / "store_performance_2018to2022.csv")
perf["invoice_date"] = pd.to_datetime(perf["invoice_date"]).dt.normalize()
perf_cols = ("invoice_count", "oc_count", "fleet_oc_count")

class DatasetBuilder:
    # many, but not all of the features that have been used for training prototypes
    location_features = list(set(store_info_cols) - {"invoice_date", "store_id"})
    calendar_features = ["year", "month", "day_of_year", "dow", "is_weekend"]
    demand_features = ["inv_lag_1", "inv_lag_7", "inv_rollmean_7", "inv_rollmean_14", "inv_rollmean_28"]
    weather_features = list(WeatherConfig.keep_cols)
    default_features = calendar_features + demand_features + weather_features

    @property
    def all_features(self):
        return location_features + calendar_features + demand_features + weather_features
    
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

        m = self.merged[["store_id", "invoice_date", *perf_cols]].copy()

        # location features
        location_cols = [c for c in features if c in self.location_features]
        if len(location_cols) != 0:
            m = m.merge(store_info[["store_id", *location_cols]], on=["store_id"], how="left")
        
        # demand features
        demand_cols = [c for c in features if c.startswith(("inv_lag_", "inv_rollmean_"))]
        m = add_demand_features(m, demand_cols)

        # weather features
        weather_cols = tuple(f for f in features if f in self.weather_features)
        weather_df = self.get_weather_df(weather_cols)
        m = m.merge(weather_df, on=["store_id", "invoice_date"], how="inner")
        m = add_weather_features(m) # includes dow, month, and is_weekend

        self.merged = m

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

    def to_csv(self, filename=None):
        if filename == None:
            filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S") + ".csv"

        train_name = "train_" + filename
        valid_name = "valid_" + filename

        if filename.endswith(".parquet"):
            self.train.to_parquet(DATA_PROCESSED / train_name, index=False)
            self.valid.to_parquet(DATA_PROCESSED / valid_name, index=False)
        else:
            self.train.to_csv(DATA_PROCESSED / train_name, index=False)
            self.valid.to_csv(DATA_PROCESSED / valid_name, index=False)
