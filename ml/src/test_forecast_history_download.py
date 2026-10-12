from pathlib import Path

import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
    / "forecast_history"
    / "event_4361_test.nc"
)


client = cdsapi.Client()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

print("NEXUS Forecast History Test")
print("---------------------------")
print()
print("Forecast date: 2016-06-12")
print("History:       2016-06-05 -> 2016-06-11")
print("Latitude:      6.933713")
print("Longitude:     -1.132254")
print()

client.retrieve(
    "reanalysis-era5-land",
    {
        "variable": [
            "2m_temperature",
            "total_precipitation",
        ],
        "year": "2016",
        "month": "06",
        "day": [
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "area": [
            7.133713,
            -1.332254,
            6.733713,
            -0.932254,
        ],
        "format": "netcdf",
    },
    str(OUTPUT_FILE),
)

print()
print("Download completed.")
print(f"Output: {OUTPUT_FILE}")