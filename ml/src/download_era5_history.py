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
    / "raw_history"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Load Sen1Floods11 metadata
# ---------------------------------------------------------

samples = []

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        samples.append(row)


# Group samples by observation date

samples_by_date = defaultdict(list)

for sample in samples:

    samples_by_date[
        sample["s1_date"]
    ].append(sample)


print("NEXUS ERA5-Land Historical Downloader")
print("======================================")
print("Total samples:", len(samples))
print("Unique observation dates:", len(samples_by_date))
print()


client = cdsapi.Client()


# ---------------------------------------------------------
# Download 7 days preceding each observation date
# ---------------------------------------------------------

for observation_date, date_samples in sorted(
    samples_by_date.items()
):

    output_file = (
        OUTPUT_DIR
        / f"era5_history_{observation_date}.nc"
    )

    if output_file.exists():

        print(
            f"[SKIP] {observation_date} "
            f"already exists."
        )

        continue


    # Geographic extent of all chips for this date

    latitudes = [
        float(sample["latitude"])
        for sample in date_samples
    ]

    longitudes = [
        float(sample["longitude"])
        for sample in date_samples
    ]


    north = min(
        90,
        max(latitudes) + 0.5
    )

    south = max(
        -90,
        min(latitudes) - 0.5
    )

    west = max(
        -180,
        min(longitudes) - 0.5
    )

    east = min(
        180,
        max(longitudes) + 0.5
    )


    # -----------------------------------------------------
    # Calculate the 7-day historical period
    # -----------------------------------------------------

    from datetime import datetime, timedelta

    observation = datetime.strptime(
        observation_date,
        "%Y-%m-%d"
    )

    start_date = observation - timedelta(
        days=7
    )

    end_date = observation - timedelta(
        days=1
    )


    years = sorted(
        {
            str(start_date.year),
            str(end_date.year),
        }
    )


    # CDS requires each date to be specified.
    dates = []

    current = start_date

    while current <= end_date:

        dates.append(
            current.strftime("%Y-%m-%d")
        )

        current += timedelta(days=1)


    years_for_request = sorted(
        set(
            date[:4]
            for date in dates
        )
    )

    months_for_request = sorted(
        set(
            date[5:7]
            for date in dates
        )
    )

    days_for_request = sorted(
        set(
            date[8:10]
            for date in dates
        )
    )


    print(
        f"[DOWNLOAD] {observation_date}"
    )

    print(
        f"  Samples: {len(date_samples)}"
    )

    print(
        f"  Historical period: "
        f"{start_date.date()} -> "
        f"{end_date.date()}"
    )

    print(
        f"  Area: "
        f"{north:.2f}, "
        f"{west:.2f}, "
        f"{south:.2f}, "
        f"{east:.2f}"
    )


    # -----------------------------------------------------
    # Important:
    #
    # Use the exact individual dates rather than a
    # Cartesian product of years/months/days.
    # -----------------------------------------------------

    request_dates = [
        date.strftime("%Y-%m-%d")
        for date in [
            start_date + timedelta(days=i)
            for i in range(7)
        ]
    ]


    # CDS ERA5-Land hourly request

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature",
                "total_precipitation",
            ],

            "year": sorted(
                set(
                    date[:4]
                    for date in request_dates
                )
            ),

            "month": sorted(
                set(
                    date[5:7]
                    for date in request_dates
                )
            ),

            "day": sorted(
                set(
                    date[8:10]
                    for date in request_dates
                )
            ),

            "time": [
                f"{hour:02d}:00"
                for hour in range(24)
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


    print(
        f"  Saved: {output_file}"
    )

    print()


print(
    "Historical ERA5-Land downloads complete."
)