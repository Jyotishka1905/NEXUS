from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_samples.csv"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
    / "forecast_history"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_features.csv"
)


def extract_features(sample, history_file):
    """Extract temporal features from D-7 to D-1 ERA5-Land history."""

    with xr.open_dataset(history_file) as ds:

        # Select the nearest ERA5 grid cell to the event/control location.
        point = ds.sel(
            latitude=sample["latitude"],
            longitude=sample["longitude"],
            method="nearest",
        )

        time = pd.to_datetime(point["valid_time"].values)

        temperature = np.asarray(point["t2m"].values, dtype=np.float64)
        precipitation = np.asarray(point["tp"].values, dtype=np.float64)

    # ERA5-Land total precipitation is stored in metres.
    precipitation_mm = precipitation * 1000.0

    temperature_c = temperature - 273.15

    # Convert hourly observations into a dataframe.
    hourly = pd.DataFrame(
        {
            "time": time,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
        }
    )

    hourly["date"] = hourly["time"].dt.date

    # Daily aggregation.
    daily = (
        hourly.groupby("date")
        .agg(
            temperature_mean_c=("temperature_c", "mean"),
            temperature_min_c=("temperature_c", "min"),
            temperature_max_c=("temperature_c", "max"),
            precipitation_sum_mm=("precipitation_mm", "sum"),
            precipitation_max_hourly_mm=("precipitation_mm", "max"),
            wet_hours=("precipitation_mm", lambda x: int((x > 1.0).sum())),
        )
        .reset_index()
    )

    # Ensure chronological order.
    daily = daily.sort_values("date").reset_index(drop=True)

    features = {
        "sample_id": sample["sample_id"],
        "label": int(sample["label"]),
        "country": sample["country"],
        "latitude": float(sample["latitude"]),
        "longitude": float(sample["longitude"]),
        "forecast_date": sample["forecast_date"],
    }

    # ---------------------------------------------------------
    # D-7 ... D-1 daily features
    # ---------------------------------------------------------

    for i, row in daily.iterrows():
        day_number = i + 7

        features[f"d{day_number}_temp_mean_c"] = row[
            "temperature_mean_c"
        ]

        features[f"d{day_number}_temp_min_c"] = row[
            "temperature_min_c"
        ]

        features[f"d{day_number}_temp_max_c"] = row[
            "temperature_max_c"
        ]

        features[f"d{day_number}_precip_mm"] = row[
            "precipitation_sum_mm"
        ]

        features[f"d{day_number}_max_hourly_precip_mm"] = row[
            "precipitation_max_hourly_mm"
        ]

        features[f"d{day_number}_wet_hours"] = row[
            "wet_hours"
        ]

    # ---------------------------------------------------------
    # Summary features
    # ---------------------------------------------------------

    precip = daily["precipitation_sum_mm"].to_numpy()

    features["precip_7day_total_mm"] = float(precip.sum())

    features["precip_max_daily_mm"] = float(precip.max())

    features["precip_last_1day_mm"] = float(precip[-1])

    features["precip_last_3days_mm"] = float(precip[-3:].sum())

    features["precip_first_3days_mm"] = float(precip[:3].sum())

    features["precip_max_3day_window_mm"] = float(
        pd.Series(precip).rolling(3).sum().max()
    )

    features["precip_wet_days"] = int((precip > 1.0).sum())

    features["precip_heavy_days"] = int((precip > 10.0).sum())

    # Simple precipitation trend:
    # positive = increasing toward the forecast date
    x = np.arange(len(precip))

    if len(precip) >= 2:
        features["precip_trend"] = float(
            np.polyfit(x, precip, 1)[0]
        )
    else:
        features["precip_trend"] = 0.0

    # Temperature summaries.
    temp = daily["temperature_mean_c"].to_numpy()

    features["temp_7day_mean_c"] = float(temp.mean())

    features["temp_last_1day_c"] = float(temp[-1])

    features["temp_last_3days_mean_c"] = float(temp[-3:].mean())

    features["temp_min_7day_c"] = float(
        daily["temperature_min_c"].min()
    )

    features["temp_max_7day_c"] = float(
        daily["temperature_max_c"].max()
    )

    return features


def main():

    print("NEXUS Future Flood Forecast Feature Builder")
    print("-------------------------------------------")

    samples = pd.read_csv(SAMPLES_FILE)

    print(f"Forecast samples: {len(samples)}")
    print()

    records = []

    failed = 0

    for index, sample in samples.iterrows():

        sample_id = sample["sample_id"]

        history_file = HISTORY_DIR / f"{sample_id}.nc"

        if not history_file.exists():
            print(f"WARNING: missing history -> {sample_id}")
            failed += 1
            continue

        try:
            features = extract_features(
                sample,
                history_file,
            )

            records.append(features)

        except Exception as exc:
            failed += 1
            print(
                f"FAILED: {sample_id} -> {exc}"
            )

        if (index + 1) % 10 == 0:
            print(
                f"Processed {index + 1}/{len(samples)}"
            )

    output = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Feature extraction complete")
    print("---------------------------")
    print(f"Successful: {len(output)}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(samples)}")
    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)

    print()
    print("Label distribution:")
    print(output["label"].value_counts().sort_index())

    print()
    print("Feature table shape:")
    print(output.shape)

    print()
    print("NaN values:")
    print(int(output.isna().sum().sum()))

    print()
    print("Infinite values:")

    numeric = output.select_dtypes(include=[np.number])

    print(
        int(
            np.isinf(numeric.to_numpy()).sum()
        )
    )


if __name__ == "__main__":
    main()