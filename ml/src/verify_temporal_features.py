from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "temporal_features.csv"
)


df = pd.read_csv(FEATURE_FILE)


print("NEXUS Temporal Feature Verification")
print("===================================")
print()

print("File:")
print(FEATURE_FILE)
print()

print("Shape:")
print(df.shape)
print()


# =========================================================
# BASIC CHECKS
# =========================================================

print("Missing values:")
print(df.isna().sum().sum())
print()

print("Infinite values:")

numeric_values = df.select_dtypes(
    include=[np.number]
).values

print(
    np.isinf(numeric_values).sum()
)

print()


# =========================================================
# TARGET
# =========================================================

print("Target distribution:")
print(
    df["target"].value_counts()
)
print()


# =========================================================
# PRECIPITATION FEATURES
# =========================================================

precipitation_columns = [
    column
    for column in df.columns
    if "precipitation_mm" in column
]


print("Precipitation statistics")
print("-------------------------")

for column in precipitation_columns:

    values = df[column].values

    print(
        f"{column}: "
        f"min={values.min():.4f}, "
        f"median={np.median(values):.4f}, "
        f"max={values.max():.4f}"
    )

print()


# =========================================================
# TEMPERATURE FEATURES
# =========================================================

temperature_columns = [
    column
    for column in df.columns
    if "temperature_mean_c" in column
]


print("Temperature statistics")
print("----------------------")

for column in temperature_columns:

    values = df[column].values

    print(
        f"{column}: "
        f"min={values.min():.2f}, "
        f"median={np.median(values):.2f}, "
        f"max={values.max():.2f}"
    )

print()


# =========================================================
# EXTREME RAINFALL SAMPLES
# =========================================================

print("Samples with highest single-day rainfall")
print("-----------------------------------------")

all_precipitation = df[
    ["chip_id", "country", "observation_date"]
    + precipitation_columns
].copy()

all_precipitation["max_daily_rainfall_mm"] = (
    all_precipitation[
        precipitation_columns
    ].max(axis=1)
)

top_samples = (
    all_precipitation
    .sort_values(
        "max_daily_rainfall_mm",
        ascending=False
    )
    .head(10)
)


for _, row in top_samples.iterrows():

    print(
        f"Chip {row['chip_id']} | "
        f"{row['country']} | "
        f"{row['observation_date']} | "
        f"Max daily rainfall: "
        f"{row['max_daily_rainfall_mm']:.2f} mm"
    )

print()


# =========================================================
# TEMPORAL FEATURE PREVIEW
# =========================================================

print("First sample")
print("------------")

first = df.iloc[0]

print(
    "Chip:",
    first["chip_id"]
)

print(
    "Country:",
    first["country"]
)

print(
    "Observation date:",
    first["observation_date"]
)

print(
    "Target:",
    first["target"]
)

print()

for day in range(1, 8):

    temperature = first[
        f"day_{day}_temperature_mean_c"
    ]

    rainfall = first[
        f"day_{day}_precipitation_mm"
    ]

    print(
        f"D-{8-day}: "
        f"Temperature={temperature:.2f} °C, "
        f"Rainfall={rainfall:.2f} mm"
    )

print()

print("Verification complete.")