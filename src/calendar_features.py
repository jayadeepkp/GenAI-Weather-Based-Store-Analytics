import pandas as pd
import holidays
from datetime import timedelta, date

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.normalize()
    out = out.sort_values(["store_id", "invoice_date"]).reset_index(drop=True)

    # generic calendar features
    out["year"] = out["invoice_date"].dt.year
    out["month"] = out["invoice_date"].dt.month
    out["day_of_year"] = out["invoice_date"].dt.dayofyear
    out["dow"] = out["invoice_date"].dt.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["time_index"] = out.groupby("store_id").cumcount()

    # holiday features - handles only federal holidays
    us_holidays = holidays.US(years=range(2018,2024)) # include 2023 holidays to account for 2022-12-31 being before New Years
    out["is_holiday"] = out["invoice_date"].dt.date.isin(us_holidays).astype(int)

    one_day = timedelta(days=1)

    tomorrow = out.invoice_date + one_day
    yesterday = out.invoice_date - one_day

    out["is_day_before_holiday"] = tomorrow.dt.date.isin(us_holidays).astype(int)
    out["is_day_after_holiday"] = yesterday.dt.date.isin(us_holidays).astype(int)

    # holiday_season feature ("holiday season" being defined as Thanksgiving to New Year's)
    season_col = pd.Series([False] * len(out["invoice_date"]))
    for year in range(2018, 2023):
        season_start = [day for day in us_holidays.get_named("Thanksgiving") if day.year == year]
        season_start = season_start[0]
        season_end = date(year+1, 1, 1)

        season_col |= out.invoice_date.dt.date.between(season_start, season_end)


    out["holiday_season"] = season_col.astype(int)

    return out
