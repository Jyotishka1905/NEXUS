from pathlib import Path
import csv

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
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
    / "environmental_features.csv"
)


def find_valid_point(ds, latitude, longitude):

    # Start with nearest grid cell
    lat_index = int(
        np.abs(
            ds.latitude.values - latitude
        ).argmin()
    )

    lon_index = int(
        np.abs(
            ds.longitude.values - longitude
        ).argmin()
    )

    # Search progressively larger neighborhoods
    max_radius = max(
        len(ds.latitude),
        len(ds.longitude),
    )

    for radius in range(
        max_radius
    ):

        lat_start = max(
            0,
            lat_index - radius,
        )

        lat_end = min(
            len(ds.latitude),
            lat_index + radius + 1,
        )

        lon_start = max(
            0,
            lon_index - radius,
        )

        lon_end = min(
            len(ds.longitude),
            lon_index + radius + 1,
        )

        for i in range(
            lat_start,
            lat_end,
        ):

            for j in range(
                lon_start,
                lon_end,
            ):

                t2m_values = (
                    ds["t2m"]
                    .isel(
                        latitude=i,
                        longitude=j,
                    )
                    .values
                )

                tp_values = (
                    ds["tp"]
                    .isel(
                        latitude=i,
                        longitude=j,
                    )
                    .values
                )

                if (
                    np.isfinite(t2m_values).all()
                    and np.isfinite(tp_values).all()
                ):

                    return (
                        float(ds.latitude.values[i]),
                        float(ds.longitude.values[j]),
                    )

    return None


samples = []

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        samples.append(row)


print("NEXUS Environmental Feature Builder")
print("===================================")
print("Samples:", len(samples))
print()


results = []

fallback_count = 0


for index, sample in enumerate(
    samples,
    start=1,
):

    chip_id = sample["chip_id"]
    date = sample["s1_date"]

    latitude = float(
        sample["latitude"]
    )

    longitude = float(
        sample["longitude"]
    )

    era5_file = (
        RAW_DIR
        / f"era5_{date}.nc"
    )

    if not era5_file.exists():

        raise FileNotFoundError(
            f"ERA5 file not found for {date}:\n"
            f"{era5_file}"
        )

    ds = xr.open_dataset(
        era5_file
    )

    # Find closest valid ERA5 grid point
    selected_point = find_valid_point(
        ds,
        latitude,
        longitude,
    )

    if selected_point is None:

        ds.close()

        raise RuntimeError(
            f"No valid ERA5 grid cell found for:\n"
            f"chip={chip_id}, "
            f"lat={latitude}, "
            f"lon={longitude}, "
            f"date={date}"
        )

    selected_latitude, selected_longitude = (
        selected_point
    )

    # Check whether fallback was required
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

    if (
        selected_latitude != nearest_latitude
        or selected_longitude != nearest_longitude
    ):

        fallback_count += 1

        print(
            f"[FALLBACK] {chip_id} "
            f"{date}: "
            f"({nearest_latitude:.1f}, "
            f"{nearest_longitude:.1f}) -> "
            f"({selected_latitude:.1f}, "
            f"{selected_longitude:.1f})"
        )

    point = ds.sel(
        latitude=selected_latitude,
        longitude=selected_longitude,
    )

    temperature_kelvin = float(
        point["t2m"].mean().values
    )

    precipitation_m = float(
        point["tp"].mean().values
    )

    temperature_celsius = (
        temperature_kelvin
        - 273.15
    )

    precipitation_mm = (
        precipitation_m
        * 1000.0
    )

    results.append(
        {
            "chip_id": chip_id,
            "country": sample["country"],
            "latitude": latitude,
            "longitude": longitude,
            "date": date,
            "temperature_c": temperature_celsius,
            "precipitation_mm": precipitation_mm,
        }
    )

    ds.close()

    if (
        index % 25 == 0
        or index == len(samples)
    ):

        print(
            f"Processed "
            f"{index}/{len(samples)} samples"
        )


fieldnames = [
    "chip_id",
    "country",
    "latitude",
    "longitude",
    "date",
    "temperature_c",
    "precipitation_mm",
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


temperatures = np.array(
    [
        row["temperature_c"]
        for row in results
    ]
)

precipitation = np.array(
    [
        row["precipitation_mm"]
        for row in results
    ]
)


print()
print(
    "Environmental feature extraction complete."
)
print("-------------------------------------------")
print("Records:", len(results))
print("Fallback grid selections:", fallback_count)
print("Output:", OUTPUT_FILE)
print()

print(
    f"Temperature range: "
    f"{temperatures.min():.2f} "
    f"to "
    f"{temperatures.max():.2f} °C"
)

print(
    f"Precipitation range: "
    f"{precipitation.min():.4f} "
    f"to "
    f"{precipitation.max():.4f} mm"
)

print(
    "NaN temperature values:",
    int(np.isnan(temperatures).sum())
)

print(
    "NaN precipitation values:",
    int(np.isnan(precipitation).sum())
)