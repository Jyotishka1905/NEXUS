from pathlib import Path
import csv
from collections import defaultdict

import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

samples = []

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        samples.append(row)


# --------------------------------------------------
# Group samples by date
# --------------------------------------------------

samples_by_date = defaultdict(list)

for sample in samples:
    samples_by_date[sample["s1_date"]].append(sample)


print("NEXUS ERA5-Land Batch Downloader")
print("================================")
print("Total samples:", len(samples))
print("Unique dates:", len(samples_by_date))
print()


client = cdsapi.Client()


# --------------------------------------------------
# Download one spatial subset per date
# --------------------------------------------------

for date, date_samples in sorted(samples_by_date.items()):

    output_file = OUTPUT_DIR / f"era5_{date}.nc"

    if output_file.exists():
        print(f"[SKIP] {date} already exists.")
        continue

    latitudes = [
        float(sample["latitude"])
        for sample in date_samples
    ]

    longitudes = [
        float(sample["longitude"])
        for sample in date_samples
    ]

    # Small margin around the Sen1Floods11 samples.
    north = min(90, max(latitudes) + 0.5)
    south = max(-90, min(latitudes) - 0.5)
    west = max(-180, min(longitudes) - 0.5)
    east = min(180, max(longitudes) + 0.5)

    year = date[:4]
    month = date[5:7]
    day = date[8:10]

    print(f"[DOWNLOAD] {date}")
    print(f"  Samples: {len(date_samples)}")
    print(
        f"  Area: "
        f"{north:.2f}, {west:.2f}, "
        f"{south:.2f}, {east:.2f}"
    )

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature",
                "total_precipitation",
            ],
            "year": year,
            "month": month,
            "day": day,
            "time": [
                "00:00",
                "06:00",
                "12:00",
                "18:00",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [
                north,
                west,
                south,
                east,
            ],
        },
        str(output_file),
    )

    print(f"  Saved: {output_file}")
    print()


print("All ERA5-Land downloads complete.")