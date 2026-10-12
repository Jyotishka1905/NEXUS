from pathlib import Path
import zipfile
import tempfile
import shutil
import time

import pandas as pd
import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "raw"
    / "forecast_history"
)


VARIABLES = [
    "2m_temperature",
    "total_precipitation",
]


TIMES = [
    f"{hour:02d}:00"
    for hour in range(24)
]


def download_sample(client, row):

    sample_id = str(row["sample_id"])

    forecast_date = pd.Timestamp(
        row["forecast_date"]
    )

    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    output_file = (
        OUTPUT_DIR
        / f"{sample_id}.nc"
    )

    if output_file.exists():

        return "skipped"

    history_start = (
        forecast_date
        - pd.Timedelta(days=7)
    )

    history_end = (
        forecast_date
        - pd.Timedelta(days=1)
    )

    dates = pd.date_range(
        history_start,
        history_end,
        freq="D",
    )

    # --------------------------------------------------------------
    # Use one exact date request.
    #
    # CDS expects year/month/day combinations, so because the
    # history can cross month/year boundaries, we group dates
    # by year/month and make one request for each month.
    # --------------------------------------------------------------

    date_groups = {}

    for date in dates:

        key = (
            date.year,
            date.month,
        )

        date_groups.setdefault(
            key,
            [],
        ).append(date.day)

    # Temporary directory for the returned ZIP/NetCDF files.
    with tempfile.TemporaryDirectory() as tmp:

        extracted_parts = []

        for (
            (year, month),
            days,
        ) in date_groups.items():

            zip_path = (
                Path(tmp)
                / f"{year}_{month:02d}.zip"
            )

            request = {
                "variable": VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [
                    f"{day:02d}"
                    for day in sorted(days)
                ],
                "time": TIMES,
                "area": [
                    latitude + 0.2,
                    longitude - 0.2,
                    latitude - 0.2,
                    longitude + 0.2,
                ],
                "format": "netcdf",
            }

            client.retrieve(
                "reanalysis-era5-land",
                request,
                str(zip_path),
            )

            # CDS returned a ZIP archive.
            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as archive:

                nc_names = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".nc")
                ]

                if not nc_names:
                    raise RuntimeError(
                        "CDS archive contains no NetCDF file."
                    )

                nc_name = nc_names[0]

                extracted = (
                    Path(tmp)
                    / f"part_{year}_{month:02d}.nc"
                )

                with archive.open(
                    nc_name
                ) as source, open(
                    extracted,
                    "wb",
                ) as target:

                    shutil.copyfileobj(
                        source,
                        target,
                    )

                extracted_parts.append(
                    extracted
                )

        # ----------------------------------------------------------
        # Most histories will be entirely within one month.
        #
        # If a history crosses a month boundary, combine the
        # monthly files with xarray.
        # ----------------------------------------------------------

        if len(extracted_parts) == 1:

            shutil.copy2(
                extracted_parts[0],
                output_file,
            )

        else:

            import xarray as xr

            datasets = [
                xr.open_dataset(
                    path
                )
                for path in extracted_parts
            ]

            try:

                combined = xr.concat(
                    datasets,
                    dim="valid_time",
                )

                combined = combined.sortby(
                    "valid_time"
                )

                combined.to_netcdf(
                    output_file
                )

            finally:

                for dataset in datasets:
                    dataset.close()

    return "downloaded"


def main():

    print()
    print(
        "NEXUS Forecast History Downloader"
    )
    print(
        "----------------------------------"
    )
    print()

    if not SAMPLES_FILE.exists():

        raise RuntimeError(
            f"Samples file not found:\n"
            f"{SAMPLES_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = pd.read_csv(
        SAMPLES_FILE
    )

    required = {
        "sample_id",
        "latitude",
        "longitude",
        "forecast_date",
        "label",
    }

    missing = (
        required
        - set(samples.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing columns: "
            + str(sorted(missing))
        )

    samples["forecast_date"] = (
        pd.to_datetime(
            samples["forecast_date"]
        )
    )

    print(
        f"Forecast samples: {len(samples)}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    client = cdsapi.Client()

    downloaded = 0
    skipped = 0
    failed = 0

    for index, row in samples.iterrows():

        sample_id = str(
            row["sample_id"]
        )

        print(
            f"[{index + 1}/{len(samples)}] "
            f"{sample_id} | "
            f"label={int(row['label'])} | "
            f"D={row['forecast_date'].date()}"
        )

        try:

            result = download_sample(
                client,
                row,
            )

            if result == "downloaded":

                downloaded += 1
                print(
                    "    -> downloaded"
                )

            else:

                skipped += 1
                print(
                    "    -> already exists"
                )

        except Exception as exc:

            failed += 1

            print(
                "    -> FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

        print()

        # Small pause between samples.
        time.sleep(1)

    print()
    print(
        "Download summary"
    )
    print(
        "----------------"
    )

    print(
        f"Downloaded: {downloaded}"
    )

    print(
        f"Skipped:    {skipped}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total:      {downloaded + skipped + failed}"
    )

    print()

    if failed:

        print(
            "Some samples failed. "
            "Run the script again to retry them."
        )

    else:

        print(
            "All forecast histories downloaded."
        )

    print()
    print("Completed.")


if __name__ == "__main__":
    main()