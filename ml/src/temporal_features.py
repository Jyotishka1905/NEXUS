from pathlib import Path
import csv
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from dataset import Sen1Floods11Dataset


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)

RAW_HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw_history"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "temporal_features.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)


# =========================================================
# FIND NEAREST VALID ERA5 GRID CELL
# =========================================================

def find_nearest_valid_point(
    ds,
    latitude,
    longitude,
):
    """
    Find the geographically nearest ERA5 grid cell where
    the complete historical temperature and precipitation
    sequences contain finite values.
    """

    latitudes = ds.latitude.values
    longitudes = ds.longitude.values

    valid_candidates = []

    for i, grid_latitude in enumerate(latitudes):

        for j, grid_longitude in enumerate(longitudes):

            temperature = (
                ds["t2m"]
                .isel(
                    latitude=i,
                    longitude=j,
                )
                .values
            )

            precipitation = (
                ds["tp"]
                .isel(
                    latitude=i,
                    longitude=j,
                )
                .values
            )

            # Require the complete historical sequence
            # to be valid.
            if (
                np.isfinite(temperature).all()
                and np.isfinite(precipitation).all()
            ):

                # Approximate geographic distance in
                # latitude/longitude degrees.
                distance = (
                    (grid_latitude - latitude) ** 2
                    +
                    (grid_longitude - longitude) ** 2
                )

                valid_candidates.append(
                    (
                        distance,
                        i,
                        j,
                    )
                )

    if not valid_candidates:
        return None

    # Select the genuinely nearest valid grid cell.
    valid_candidates.sort(
        key=lambda item: item[0]
    )

    _, best_i, best_j = valid_candidates[0]

    return (
        float(latitudes[best_i]),
        float(longitudes[best_j]),
    )


# =========================================================
# LOAD SEN1FLOODS11 METADATA
# =========================================================

samples = []

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        samples.append(row)


print("NEXUS Temporal Feature Builder")
print("==============================")
print("Samples:", len(samples))
print()


# =========================================================
# LOAD SEN1FLOODS11 DATASET
# =========================================================

dataset = Sen1Floods11Dataset(
    DATA_ROOT
)

path_to_index = {
    str(image_path): index
    for index, (image_path, _) in enumerate(
        dataset.samples
    )
}


# =========================================================
# PROCESS EACH CHIP
# =========================================================

records = []

fallback_count = 0


for index, sample in enumerate(
    samples,
    start=1,
):

    chip_id = sample["chip_id"]

    observation_date = sample["s1_date"]

    latitude = float(
        sample["latitude"]
    )

    longitude = float(
        sample["longitude"]
    )


    # -----------------------------------------------------
    # Historical ERA5 file
    # -----------------------------------------------------

    history_file = (
        RAW_HISTORY_DIR
        / f"era5_history_{observation_date}.nc"
    )

    if not history_file.exists():

        raise FileNotFoundError(
            f"Historical ERA5 file not found:\n"
            f"{history_file}"
        )


    # -----------------------------------------------------
    # Open ERA5 dataset
    # -----------------------------------------------------

    ds = xr.open_dataset(
        history_file
    )


    # -----------------------------------------------------
    # Find nearest valid grid cell
    # -----------------------------------------------------

    selected_point = find_nearest_valid_point(
        ds,
        latitude,
        longitude,
    )

    if selected_point is None:

        ds.close()

        raise RuntimeError(
            f"No valid ERA5 grid cell found for:\n"
            f"chip={chip_id}\n"
            f"date={observation_date}\n"
            f"latitude={latitude}\n"
            f"longitude={longitude}"
        )


    selected_latitude, selected_longitude = (
        selected_point
    )


    # -----------------------------------------------------
    # Determine nearest grid cell
    # -----------------------------------------------------

    nearest_latitude = float(
        ds.latitude.values[
            np.abs(
                ds.latitude.values
                - latitude
            ).argmin()
        ]
    )

    nearest_longitude = float(
        ds.longitude.values[
            np.abs(
                ds.longitude.values
                - longitude
            ).argmin()
        ]
    )


    # -----------------------------------------------------
    # Record fallback
    # -----------------------------------------------------

    if (
        selected_latitude != nearest_latitude
        or selected_longitude != nearest_longitude
    ):

        fallback_count += 1

        fallback_distance = np.sqrt(
            (
                selected_latitude
                - latitude
            ) ** 2
            +
            (
                selected_longitude
                - longitude
            ) ** 2
        )

        print(
            f"[FALLBACK] "
            f"{chip_id} "
            f"{observation_date}: "
            f"({nearest_latitude:.1f}, "
            f"{nearest_longitude:.1f}) "
            f"-> "
            f"({selected_latitude:.1f}, "
            f"{selected_longitude:.1f}) "
            f"distance={fallback_distance:.3f}°"
        )


    # -----------------------------------------------------
    # Extract selected grid cell
    # -----------------------------------------------------

    point = ds.sel(
        latitude=selected_latitude,
        longitude=selected_longitude,
    )


    temperature_kelvin = (
        point["t2m"].values
    )

    precipitation_m = (
        point["tp"].values
    )


    # Convert units
    temperature_celsius = (
        temperature_kelvin
        - 273.15
    )

    precipitation_mm = (
        precipitation_m
        * 1000.0
    )


    # -----------------------------------------------------
    # Convert ERA5 timestamps
    # -----------------------------------------------------

    times = pd.to_datetime(
        ds.valid_time.values
    )


    temporal_df = pd.DataFrame(
        {
            "time": times,
            "temperature_c": temperature_celsius,
            "precipitation_mm": precipitation_mm,
        }
    )


    # Normalize each timestamp to its calendar day.
    temporal_df["date"] = (
        temporal_df["time"]
        .dt.normalize()
    )


    # -----------------------------------------------------
    # Explicitly define expected historical dates
    # -----------------------------------------------------

    observation = datetime.strptime(
        observation_date,
        "%Y-%m-%d"
    )


    expected_dates = [
        observation - timedelta(days=offset)
        for offset in range(7, 0, -1)
    ]


    expected_dates = pd.to_datetime(
        expected_dates
    )


    # -----------------------------------------------------
    # Keep ONLY the seven expected historical days
    # -----------------------------------------------------

    temporal_df = temporal_df[
        temporal_df["date"].isin(
            expected_dates
        )
    ].copy()


    # -----------------------------------------------------
    # Verify exactly 7 days
    # -----------------------------------------------------

    available_dates = sorted(
        temporal_df["date"]
        .unique()
    )


    if len(available_dates) != 7:

        ds.close()

        raise RuntimeError(
            f"Expected 7 historical days but found "
            f"{len(available_dates)} for chip "
            f"{chip_id}.\n"
            f"Expected dates:\n"
            f"{expected_dates}\n"
            f"Available dates:\n"
            f"{available_dates}"
        )


    # -----------------------------------------------------
    # Verify 24 hourly observations per day
    # -----------------------------------------------------

    hours_per_day = (
        temporal_df
        .groupby("date")
        .size()
    )


    if not (
        hours_per_day == 24
    ).all():

        ds.close()

        raise RuntimeError(
            f"Expected 24 hourly observations per day "
            f"for chip {chip_id}.\n"
            f"Hourly counts:\n"
            f"{hours_per_day}"
        )


    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    temporal_df = (
        temporal_df
        .sort_values("date")
        .reset_index(drop=True)
    )


    # -----------------------------------------------------
    # Aggregate hourly ERA5 data to daily features
    # -----------------------------------------------------

    daily = (
        temporal_df
        .groupby("date")
        .agg(
            temperature_mean_c=(
                "temperature_c",
                "mean",
            ),
            temperature_min_c=(
                "temperature_c",
                "min",
            ),
            temperature_max_c=(
                "temperature_c",
                "max",
            ),
            precipitation_mm=(
                "precipitation_mm",
                "sum",
            ),
        )
        .reset_index()
    )


    # -----------------------------------------------------
    # Final daily verification
    # -----------------------------------------------------

    if len(daily) != 7:

        ds.close()

        raise RuntimeError(
            f"Daily aggregation produced "
            f"{len(daily)} days instead of 7 "
            f"for chip {chip_id}"
        )


    # -----------------------------------------------------
    # Determine flood target
    #
    # Target = 1 when >=1% of valid pixels are labelled
    # as water/flood in the Sen1Floods11 hand label.
    # -----------------------------------------------------

    image_path = next(
        (
            path
            for path, _ in dataset.samples
            if chip_id in path.name
        ),
        None,
    )


    if image_path is None:

        ds.close()

        raise RuntimeError(
            f"Could not find image for chip "
            f"{chip_id}"
        )


    dataset_index = path_to_index[
        str(image_path)
    ]


    _, label = dataset[
        dataset_index
    ]


    valid = label != -1


    if valid.sum().item() == 0:

        flood_fraction = 0.0

    else:

        flood_fraction = (
            (label[valid] == 1)
            .sum()
            .item()
            /
            valid.sum().item()
        )


    target = (
        1
        if flood_fraction >= 0.01
        else 0
    )


    # -----------------------------------------------------
    # Create one row per chip
    # -----------------------------------------------------

    record = {
        "chip_id": chip_id,
        "country": sample["country"],
        "latitude": latitude,
        "longitude": longitude,
        "observation_date": observation_date,
        "target": target,
    }


    # -----------------------------------------------------
    # Add 7 chronological daily timesteps
    #
    # day_1 = D-7
    # day_2 = D-6
    # ...
    # day_7 = D-1
    # -----------------------------------------------------

    for day_index, row in daily.iterrows():

        day_number = day_index + 1


        record[
            f"day_{day_number}_temperature_mean_c"
        ] = float(
            row["temperature_mean_c"]
        )


        record[
            f"day_{day_number}_temperature_min_c"
        ] = float(
            row["temperature_min_c"]
        )


        record[
            f"day_{day_number}_temperature_max_c"
        ] = float(
            row["temperature_max_c"]
        )


        record[
            f"day_{day_number}_precipitation_mm"
        ] = float(
            row["precipitation_mm"]
        )


    records.append(record)


    ds.close()


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if (
        index % 25 == 0
        or index == len(samples)
    ):

        print(
            f"Processed "
            f"{index}/{len(samples)} samples"
        )


# =========================================================
# CREATE DATAFRAME
# =========================================================

output_df = pd.DataFrame(
    records
)


# =========================================================
# SAVE
# =========================================================

output_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# VALIDATION
# =========================================================

feature_columns = [
    column
    for column in output_df.columns
    if column.startswith("day_")
]


values = (
    output_df[
        feature_columns
    ]
    .astype(float)
    .values
)


print()
print(
    "Temporal feature extraction complete."
)
print("---------------------------------------")
print(
    "Records:",
    len(output_df)
)
print(
    "Columns:",
    len(output_df.columns)
)
print(
    "Output:",
    OUTPUT_FILE
)
print()


print(
    "Temporal features per sample:",
    len(feature_columns)
)


print(
    "Total temporal feature values:",
    values.size
)


print(
    "NaN values:",
    int(
        np.isnan(values).sum()
    )
)


print(
    "Infinite values:",
    int(
        np.isinf(values).sum()
    )
)


print()


print(
    "Target distribution:"
)

print(
    output_df[
        "target"
    ].value_counts()
)


print()


print(
    "Fallback grid selections:",
    fallback_count
)


print()


# =========================================================
# SHOW FIRST SAMPLE
# =========================================================

print()
print("First sample temporal sequence:")
print("--------------------------------")

first = output_df.iloc[0]

for day_number in range(1, 8):

    temperature = first[
        f"day_{day_number}_temperature_mean_c"
    ]

    rainfall = first[
        f"day_{day_number}_precipitation_mm"
    ]

    print(
        f"Day {day_number}: "
        f"Temperature={temperature:.2f} °C, "
        f"Rainfall={rainfall:.4f} mm"
    )