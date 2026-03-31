# Prediction API Contract

## How to Run
First, make sure you have the required dependency `fastapi[standard]`:

	pip install fastapi[standard]
	
The API module also needs a few files to run:
- **`data_processed/predictions_hgb_hierarchical_2022.csv`**  
  The output of
  `notebooks/07_sprint2_hierarchical_oc_weather_intervals.ipynb`. It
  includes previous model predictions for 2022.
- **`data_processed/model_table.parquet`** or
  **`data_processed/valid_model_table.parquet`**  
  The output of `scripts/build_dataset.py`. It provides model-ready
  data for training and validation. The API requires only the
  validation set so that it can engineer features for dates after 2022.
  
With these requirements satisfied, you can run the API using the
following command:

    fastapi dev <path_to_api_dir>
	
The path argument is unneeded if this is executed within `api/`.
	
## Endpoints
### POST /predict
#### Request
This endpoint takes no parameters. The body must be formatted in JSON
like so:

**Main Body**

| Field     | Type                                       | Requried?                  | Notes                                    |
|-----------|--------------------------------------------|----------------------------|------------------------------------------|
| store\_id | int                                        | Yes                        |                                          |
| date      | date                                       | Yes                        | Must not be before 2022.                 |
| weather   | WeatherForecast object (see below) or null | Only if date is after 2022 | Ignored if prediction is already cached. |

**WeatherForecast object**

| Field | Type  | Required? | Notes       |
|-------|-------|-----------|-------------|
| tmin  | float | Yes       | Celsius     |
| tmax  | float | Yes       | Celsius     |
| tavg  | float | Yes       | Celsius     |
| prcp  | float | Yes       | Millimeters |
| wspd  | float | Yes       | Km/h        |
| snow  | float | Yes       | Millimeters |

**Example requests**

``` json
{
	store_id: 000000,
	date: "2022-11-02"
}
```

``` json
{
	"store_id": 000000,
	"date": "2024-03-09",
	"weather": {
		"tmin": 5,
		"tmax": 9.4,
		"tavg": 7,
		"prcp": 1.4,
		"wspd": 20,
		"snow": 0
	}
}
```

#### Responses
**200 OK**

A successful response will be formatted like so:

| Field           | Type          |
|-----------------|---------------|
| store\_id       | int           |
| date            | string        |
| pred\_invoice   | float         |
| pred\_oc        | float         |
| oc\_lo95        | float         |
| oc\_hi95        | float         |
| severity        | string        |
| oc\_pchg        | float         |
| actual\_invoice | float or null |
| actual\_pred    | float or null |

A (fake) example response:

``` json
{
	"store_id": 000000,
	"date": "2022-03-09",
	"pred_invoice": 12.9999991,
	"pred_oc": 8.123123123123,
	"oc_lo95": 6.0502047071876,
	"oc_hi95": 10.0000135,
	"severity": "sev_freezing",
	"oc_pchg": -5.121561532056,
	"actual_invoice": 13,
	"actual_oc": 7
}
```

**400 Bad Request**

There are several reasons why you might receive this error, including:
* `store_id` is not a valid store id
* `date` is before 2022
* `weather` was not provided, and the prediction has not been cached.
