from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECASTING_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecasting_features.csv"
)


def main():
    print("NEXUS Forecasting Dataset Verification")
    print("=======================================")
    print()

    if not FORECASTING_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found:\n{FORECASTING_FILE}"
        )

    df = pd.read_csv(FORECASTING_FILE)

    print("File:", FORECASTING_FILE)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    required_columns = [
        "sample_id",
        "latitude",
        "longitude",
        "target_date",
        "target",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    print("Required columns: OK")

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    print()
    print("Target distribution")
    print("-------------------")

    print(df["target"].value_counts().sort_index())

    unique_targets = sorted(df["target"].dropna().unique().tolist())

    if not set(unique_targets).issubset({0, 1}):
        raise RuntimeError(
            f"Target must contain only 0 and 1. Found: {unique_targets}"
        )

    if len(unique_targets) < 2:
        raise RuntimeError(
            "Dataset contains only one target class."
        )

    print("Binary target: OK")

    # ---------------------------------------------------------
    # Duplicate samples
    # ---------------------------------------------------------

    print()
    print("Duplicate check")
    print("---------------")

    duplicates = df.duplicated(
        subset=[
            "sample_id",
            "latitude",
            "longitude",
            "target_date",
            "target",
        ]
    ).sum()

    print("Duplicate rows:", duplicates)

    if duplicates > 0:
        raise RuntimeError(
            "Duplicate forecasting samples detected."
        )

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    print()
    print("Date check")
    print("----------")

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    invalid_dates = df["target_date"].isna().sum()

    print("Invalid target dates:", invalid_dates)

    if invalid_dates > 0:
        raise RuntimeError(
            "Invalid target dates detected."
        )

    print(
        "Target date range:",
        df["target_date"].min().date(),
        "to",
        df["target_date"].max().date(),
    )

    print("Unique target dates:", df["target_date"].nunique())

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    print()
    print("Coordinate check")
    print("----------------")

    invalid_latitude = (
        (~df["latitude"].between(-90, 90))
        | df["latitude"].isna()
    ).sum()

    invalid_longitude = (
        (~df["longitude"].between(-180, 180))
        | df["longitude"].isna()
    ).sum()

    print("Invalid latitudes:", invalid_latitude)
    print("Invalid longitudes:", invalid_longitude)

    if invalid_latitude > 0 or invalid_longitude > 0:
        raise RuntimeError(
            "Invalid geographic coordinates detected."
        )

    # ---------------------------------------------------------
    # Numerical features
    # ---------------------------------------------------------

    metadata_columns = [
        "sample_id",
        "latitude",
        "longitude",
        "target_date",
        "target",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print()
    print("Feature check")
    print("-------------")

    print("Feature count:", len(feature_columns))

    if len(feature_columns) == 0:
        raise RuntimeError(
            "No forecasting features found."
        )

    feature_values = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    nan_count = int(feature_values.isna().sum().sum())

    infinite_count = int(
        np.isinf(feature_values.to_numpy()).sum()
    )

    print("NaN values:", nan_count)
    print("Infinite values:", infinite_count)

    if nan_count > 0:
        raise RuntimeError(
            "Forecasting features contain NaN values."
        )

    if infinite_count > 0:
        raise RuntimeError(
            "Forecasting features contain infinite values."
        )

    # ---------------------------------------------------------
    # Sample summary
    # ---------------------------------------------------------

    print()
    print("Dataset summary")
    print("----------------")

    print("Total samples:", len(df))
    print(
        "Flood samples:",
        int((df["target"] == 1).sum()),
    )
    print(
        "Non-flood samples:",
        int((df["target"] == 0).sum()),
    )

    print(
        "Unique locations:",
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .shape[0],
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("Forecasting dataset verification complete.")
    print("Dataset structure is valid for the next experiment.")


if __name__ == "__main__":
    main()