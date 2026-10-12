from pathlib import Path
import csv
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

OUTPUT_FILE = OUTPUT_DIR / "era5_Bolivia_103757.nc"


# --------------------------------------------------
# Read first NEXUS sample
# --------------------------------------------------

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    sample = next(reader)


chip_id = sample["chip_id"]
latitude = float(sample["latitude"])
longitude = float(sample["longitude"])
date = sample["s1_date"]

year = date[:4]
month = date[5:7]
day = date[8:10]


print("NEXUS ERA5-Land Sample Extraction")
print("=================================")
print("Chip ID:", chip_id)
print("Latitude:", latitude)
print("Longitude:", longitude)
print("Date:", date)
print()


# --------------------------------------------------
# Download ERA5-Land data
# --------------------------------------------------

client = cdsapi.Client()

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
            latitude + 0.1,
            longitude - 0.1,
            latitude - 0.1,
            longitude + 0.1,
        ],
    },
    str(OUTPUT_FILE),
)


print()
print("ERA5-Land extraction complete.")
print("Saved to:", OUTPUT_FILE)