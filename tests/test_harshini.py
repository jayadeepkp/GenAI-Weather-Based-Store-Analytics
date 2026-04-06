import os
import pandas as pd


def test_weather_definitively_impacts_business():
    """
    Real User Case:
    Valvoline needs a definitive YES or NO answer on whether weather
    impacts store visits, with statistical proof leadership can trust.

    Project Requirement:
    Main goal is to answer YES or NO on weather impact with confidence.
    95% confidence intervals required.
    """

    impact_csv = "notebooks/valvoline_evaluation/weather_impact_oc.csv"

    # File must exist
    assert os.path.exists(impact_csv), (
        "Impact CSV not found. Run the evaluation notebook first."
    )

    impact = pd.read_csv(impact_csv)

    required_cols = [
        "Weather Condition",
        "N Days",
        "Mean % vs Baseline",
        "95% CI Low",
        "95% CI High",
        "Significant?",
    ]
    for col in required_cols:
        assert col in impact.columns, f"Missing required column: {col}"

    # Must have sufficient sample size and valid confidence intervals
    for _, row in impact.iterrows():
        assert row["N Days"] >= 100, (
            f"Too few days for {row['Weather Condition']}: "
            f"{row['N Days']} — not statistically credible"
        )
        assert row["95% CI Low"] < row["Mean % vs Baseline"], (
            f"Invalid CI lower bound for {row['Weather Condition']}"
        )
        assert row["Mean % vs Baseline"] < row["95% CI High"], (
            f"Invalid CI upper bound for {row['Weather Condition']}"
        )

    # CORE BUSINESS FINDING 1 — Heavy rain hurts business
    heavy_rain_rows = impact[
        impact["Weather Condition"].str.contains("Heavy Rain", case=False, na=False)
    ]
    assert not heavy_rain_rows.empty, "Heavy Rain row not found in impact CSV"
    heavy_rain = heavy_rain_rows.iloc[0]

    assert heavy_rain["Mean % vs Baseline"] < -2.0, (
        f"Heavy rain impact too small: {heavy_rain['Mean % vs Baseline']:.2f}%"
    )
    assert heavy_rain["Significant?"] == "YES", (
        "Heavy rain must be statistically significant"
    )
    assert heavy_rain["95% CI High"] < 0, (
        "Entire heavy rain CI must be below zero"
    )

    # CORE BUSINESS FINDING 2 — Snow impacts business
    snow_rows = impact[
        impact["Weather Condition"].str.contains("Snow", case=False, na=False)
    ]
    assert not snow_rows.empty, "Snow condition row not found in impact CSV"
    snow = snow_rows.iloc[0]

    assert snow["Significant?"] == "YES", (
        "At least one snow-related condition must be statistically significant"
    )

    # CORE BUSINESS FINDING 3 — Day after heavy rain is present
    day_after_rain_rows = impact[
        impact["Weather Condition"].str.contains(
            "Day After Heavy Rain", case=False, na=False
        )
    ]
    assert not day_after_rain_rows.empty, (
        "Day After Heavy Rain row not found in impact CSV"
    )
    day_after_rain = day_after_rain_rows.iloc[0]

    # This test checks that the post-rain condition is explicitly evaluated
    assert day_after_rain["N Days"] >= 100, (
        "Day After Heavy Rain does not have enough data"
    )

    # Count significant conditions
    significant = impact[impact["Significant?"] == "YES"]
    assert len(significant) >= 5, (
        f"Only {len(significant)} significant conditions — expected at least 5"
    )

    print("Business conclusion: YES — weather impacts OC visits")
    print(
        f"   Heavy rain: {heavy_rain['Mean % vs Baseline']:.2f}% — SIGNIFICANT"
    )
    print(
        f"   Snow condition: {snow['Weather Condition']} = "
        f"{snow['Mean % vs Baseline']:.2f}% — {snow['Significant?']}"
    )
    print(
        f"   Day after heavy rain included with {day_after_rain['N Days']} days"
    )
    print(f"   Total significant conditions: {len(significant)}")