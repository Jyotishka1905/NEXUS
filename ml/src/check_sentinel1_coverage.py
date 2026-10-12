from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_samples.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "sentinel1_forecast_coverage.csv"
)

CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)


def query_sentinel1(latitude, longitude, forecast_date):

    forecast_date = pd.Timestamp(forecast_date)

    # Only information available BEFORE the forecast date.
    start = (
        forecast_date - pd.Timedelta(days=7)
    ).strftime("%Y-%m-%dT00:00:00.000Z")

    end = (
        forecast_date + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%dT00:00:00.000Z")

    point = (
        f"geography'SRID=4326;"
        f"POINT({longitude} {latitude})'"
    )

    filter_expression = (
        "Collection/Name eq 'SENTINEL-1' "
        "and "
        f"OData.CSC.Intersects(area={point}) "
        "and "
        "Attributes/OData.CSC.StringAttribute/"
        "any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value "
        "eq 'IW_GRDH_1S') "
        f"and ContentDate/Start ge {start} "
        f"and ContentDate/Start lt {end}"
    )

    params = {
        "$filter": filter_expression,
        "$orderby": "ContentDate/Start desc",
        "$top": 20,
        "$select": (
            "Id,Name,ContentDate,GeoFootprint,S3Path"
        ),
    }

    response = requests.get(
        CATALOGUE_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json().get("value", [])


def main():

    print("NEXUS Forecast-Valid Sentinel-1 Coverage")
    print("=========================================")

    samples = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["forecast_date"],
    )

    print(
        f"Forecast samples: {len(samples)}"
    )
    print()
    print(
        "Search window: D-7 through D"
    )
    print(
        "Future satellite observations: EXCLUDED"
    )
    print()

    records = []

    for index, sample in samples.iterrows():

        sample_id = sample["sample_id"]

        try:

            products = query_sentinel1(
                sample["latitude"],
                sample["longitude"],
                sample["forecast_date"],
            )

            forecast_date = pd.Timestamp(
                sample["forecast_date"]
            )

            valid_products = []

            for product in products:

                acquisition = pd.Timestamp(
                    product["ContentDate"]["Start"]
                )

                # Remove timezone for comparison.
                acquisition_naive = (
                    acquisition.tz_localize(None)
                    if acquisition.tzinfo is not None
                    else acquisition
                )

                if acquisition_naive <= forecast_date:
                    valid_products.append(
                        (product, acquisition_naive)
                    )

            if valid_products:

                # Closest acquisition BEFORE the forecast.
                best, acquisition = min(
                    valid_products,
                    key=lambda item:
                    abs(
                        (
                            item[1]
                            - forecast_date
                        ).total_seconds()
                    ),
                )

                difference_days = (
                    forecast_date - acquisition
                ).total_seconds() / 86400.0

                records.append(
                    {
                        "sample_id": sample_id,
                        "label": int(sample["label"]),
                        "country": sample["country"],
                        "latitude": float(sample["latitude"]),
                        "longitude": float(sample["longitude"]),
                        "forecast_date": forecast_date.date(),
                        "coverage": 1,
                        "product_name": best["Name"],
                        "product_id": best["Id"],
                        "acquisition_date": acquisition,
                        "date_difference_days": difference_days,
                    }
                )

                print(
                    f"[{index + 1}/{len(samples)}] "
                    f"{sample_id}: FOUND "
                    f"({difference_days:.2f} days before)"
                )

            else:

                records.append(
                    {
                        "sample_id": sample_id,
                        "label": int(sample["label"]),
                        "country": sample["country"],
                        "latitude": float(sample["latitude"]),
                        "longitude": float(sample["longitude"]),
                        "forecast_date": forecast_date.date(),
                        "coverage": 0,
                    }
                )

                print(
                    f"[{index + 1}/{len(samples)}] "
                    f"{sample_id}: NONE"
                )

        except Exception as exc:

            print(
                f"[{index + 1}/{len(samples)}] "
                f"{sample_id}: ERROR -> {exc}"
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "label": int(sample["label"]),
                    "country": sample["country"],
                    "latitude": float(sample["latitude"]),
                    "longitude": float(sample["longitude"]),
                    "forecast_date": forecast_date.date(),
                    "coverage": -1,
                }
            )

    result = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Coverage summary")
    print("----------------")

    print(
        result["coverage"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Covered samples:",
        int((result["coverage"] == 1).sum()),
    )

    print(
        "No coverage:",
        int((result["coverage"] == 0).sum()),
    )

    print(
        "Errors:",
        int((result["coverage"] == -1).sum()),
    )

    covered = result[
        result["coverage"] == 1
    ]

    if not covered.empty:

        print()
        print("Acquisition timing:")
        print(
            covered[
                "date_difference_days"
            ].describe()
        )

        print()
        print("Coverage by country:")

        print(
            result.groupby("country")["coverage"]
            .agg(
                samples="count",
                covered=lambda x:
                    int((x == 1).sum()),
            )
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()