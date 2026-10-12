from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_samples.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_satellite_matches.csv"
)


# Matching tolerances.
# Satellite observations and DFO representative points will not
# generally have identical coordinates, so use a spatial tolerance.
MAX_DISTANCE_DEGREES = 0.25
MAX_DATE_DAYS = 3


def main():

    print("NEXUS Forecast Satellite Matching")
    print("=================================")

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["forecast_date"],
    )

    satellite = pd.read_csv(
        METADATA_FILE,
        parse_dates=["datetime", "s1_date"],
    )

    print(f"Forecast samples: {len(forecast)}")
    print(f"Satellite samples: {len(satellite)}")

    matches = []

    for _, sample in forecast.iterrows():

        sample_lat = float(sample["latitude"])
        sample_lon = float(sample["longitude"])
        forecast_date = sample["forecast_date"]

        candidates = satellite.copy()

        # Date difference.
        candidates["date_difference_days"] = (
            candidates["s1_date"] - forecast_date
        ).abs().dt.days

        candidates = candidates[
            candidates["date_difference_days"]
            <= MAX_DATE_DAYS
        ]

        if candidates.empty:
            continue

        # Spatial difference in degrees.
        candidates["spatial_distance"] = (
            (candidates["latitude"] - sample_lat) ** 2
            + (candidates["longitude"] - sample_lon) ** 2
        ) ** 0.5

        candidates = candidates[
            candidates["spatial_distance"]
            <= MAX_DISTANCE_DEGREES
        ]

        if candidates.empty:
            continue

        # Combined matching score.
        candidates["match_score"] = (
            candidates["spatial_distance"]
            + 0.01
            * candidates["date_difference_days"]
        )

        best = candidates.sort_values(
            "match_score"
        ).iloc[0]

        matches.append(
            {
                "sample_id": sample["sample_id"],
                "label": int(sample["label"]),
                "forecast_country": sample["country"],
                "forecast_latitude": sample_lat,
                "forecast_longitude": sample_lon,
                "forecast_date": forecast_date.date(),

                "satellite_chip_id": best["chip_id"],
                "satellite_country": best["country"],
                "satellite_latitude": best["latitude"],
                "satellite_longitude": best["longitude"],
                "satellite_date": best["s1_date"].date(),

                "date_difference_days": int(
                    best["date_difference_days"]
                ),

                "spatial_distance_degrees": float(
                    best["spatial_distance"]
                ),
            }
        )

    result = pd.DataFrame(matches)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Matching complete")
    print("-----------------")
    print(f"Matches: {len(result)}")
    print(f"Unmatched: {len(forecast) - len(result)}")

    if len(result) > 0:

        print()
        print("Matches by label:")
        print(
            result["label"]
            .value_counts()
            .sort_index()
        )

        print()
        print("Matches by country:")
        print(
            result["forecast_country"]
            .value_counts()
        )

        print()
        print("Date difference:")
        print(
            result["date_difference_days"]
            .describe()
        )

        print()
        print("Spatial distance:")
        print(
            result["spatial_distance_degrees"]
            .describe()
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()