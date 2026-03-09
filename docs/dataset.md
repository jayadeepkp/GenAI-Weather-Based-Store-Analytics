# dataset.py Module Documentation

## Constants
### store_info
The store info table.

### perf
The store performance table.

## class DatasetBuilder
### Class Variables
#### default\_features
The default features to be used in a DatasetBuilder object. Its value
is equal to the features in `reports/metrics_hgb_hierarchical.json`,
with some name slight name changes to integrate better with other
code. See below:

```python
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
```

### Instance Variables
#### cutoff
The date specifying the cutoff between training and validation
data subsets.

#### store_ids
The IDs of the stores included in the current dataset.

#### features
A list of the feature columns in the dataset. Setting the value after
initialization automatically changes the feature columns in the dataset.

#### merged
The full dataset before being partitioned into training and validation
datasets.

#### train
The training portion of the dataset. It includes entries where
`invoice_date` < `cutoff`.

#### valid
The validation portion of the dataset. It includes entries where
`invoice_date` >= `cutoff`.

### Methods
#### DatasetBuilder(*, cutoff="2022-01-01", features=default_features, n_stores=None, store_selectors={})
The DatasetBuilder constructor.

**Parameters:**
- **cutoff : *str or datetime, default "2022-01-01"***  
  The date that splits the dataset into training/validation sets.  
  Training dates <= cutoff  
  Validation dates > cutoff
- **features : *sequence of str, default default_features***  
  The list of features to include in the dataset. It can include any
  of the features listed in `default_features`, as well as some
  others:
  - Any column in `store_info`
  - Any custom lag/rolling feature, so long as it's in the format
    `<base_feature>_<lag | rollmean>_<n_days>`
  - Holiday features: `is_holiday`, `is_day_before_holiday`,
    `is_day_after_holiday`, `holiday_season`
- **n_stores : *int or None, default None***  
  Number of stores to include in the dataset.  
  `None` indicates that the method should consider all stores.
- **store_selectors : *dict with items of type [string, Any]***  
  Chose which stores to include in the dataset using `store_info`
  column names as keys. Make sure the value types line up with the
  type in its respective column. Values can be a single value or a
  sequence of values.  
  `{}` indicates the method should consider all stores.

#### select_stores(n_stores=None, columns={})
Limit the dataset to only use specific stores.

**Parameters:**
- **n_stores : *int or None, default None***  
  Number of stores to include in the dataset.  
  `None` indicates that the method should consider all stores.
- **columns : *dict with items of type [string, Any], default {}***  
  Chose which stores to include in the dataset using `store_info`
  column names as keys. Make sure the value types line up with the
  type in its respective column. Values can be a single value or a
  sequence of values.  
  `{}` indicates the method should consider all stores.
  
#### get_weather_df(cols=("tavg", "tmin", "tmax", "prcp", "snow", "wspd"))
Retrieve the weather data for each entry in the dataset. This only
retrieves the base features supplied by Meteostat.

**Parameters:**
- **cols : *tuple of str, default `("tavg", "tmin", "tmax", "prcp",
  "snow", "wspd")`***  
  Specify which base weather columns to include in the
  
**Returns:**  
A dataframe with columns `store_id`, `invoice_date`, and the coulumns
selected in `cols`.
  
#### to_file(filename=None)
Output training and validation datasets to separate files.

**Parameters:**
- **filename : *str or None, default None***  
  The name of the file to output to. The actual names of the
  training/validation datasests will be `"train_" + filename` and
  `"valid_" + filename`, respectively. Can output either a `.parquet`
  file or a `.csv` file, depending on the name given.  
  `None` will set `filename` to the current timestamp, and will output
  a `.parquet` file.

#### to_csv(filename=None)
Output training and validation datasets to separate  csv files.

**Parameters:**
- **filename : *str or None, default None***  
  The name of the file to output to. The actual names of the
  training/validation datasests will be `"train_" + filename` and
  `"valid_" + filename`, respectively.
  `None` will set `filename` to the current timestamp.

#### to_csv(filename=None)
Output training and validation datasets to separate parquet files.

**Parameters:**
- **filename : *str or None, default None***  
  The name of the file to output to. The actual names of the
  training/validation datasests will be `"train_" + filename` and
  `"valid_" + filename`, respectively.
  `None` will set `filename` to the current timestamp.
