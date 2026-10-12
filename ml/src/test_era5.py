from pathlib import Path
import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "environmental" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "era5_test.nc"


client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-land",
    {
        "variable": [
            "2m_temperature",
            "total_precipitation",
        ],
        "year": "2020",
        "month": "01",
        "day": "01",
        "time": "12:00",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [23, 87, 22, 88],
    },
    str(OUTPUT_FILE),
)

print()
print("ERA5-Land test download complete.")
print("Saved to:", OUTPUT_FILE)