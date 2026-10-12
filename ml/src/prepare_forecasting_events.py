from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "flood_events_expanded.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "forecasting_events.csv"
)


RELEVANT_COUNTRIES = [
    "India",
    "Sri Lanka",
    "Ghana",
]


def event_includes_country(country_field, country):
    """
    Check whether a DFO country field contains the country.

    Examples:
        India -> India
        India • Bangladesh -> India
        Bangladesh • India -> India
    """
    countries = [
        item.strip()
        for item in str(country_field).split("•")
    ]

    return country in countries


def main():

    print("NEXUS Forecasting Event Preparation")
    print("====================================")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input catalogue not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Input events:", len(df))
    print()

    # Normalize date columns.
    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce",
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce",
    )

    if df["start_date"].isna().any():
        raise RuntimeError(
            "Invalid start dates detected."
        )

    if df["end_date"].isna().any():
        raise RuntimeError(
            "Invalid end dates detected."
        )

    # Determine which relevant countries each event covers.
    selected_rows = []

    for _, row in df.iterrows():

        matched_countries = []

        for country in RELEVANT_COUNTRIES:
            if event_includes_country(
                row["country"],
                country,
            ):
                matched_countries.append(country)

        if not matched_countries:
            continue

        for country in matched_countries:

            selected_rows.append(
                {
                    "event_id": str(row["event_id"]),
                    "dfo_country": str(row["country"]),
                    "country": country,
                    "start_date": row["start_date"].date(),
                    "end_date": row["end_date"].date(),
                    "area_km2": row["area_km2"],
                    "target_source": row["target_source"],
                }
            )

    events = pd.DataFrame(selected_rows)

    if events.empty:
        raise RuntimeError(
            "No relevant forecasting events were found."
        )

    # Remove accidental duplicates.
    events = events.drop_duplicates(
        subset=[
            "event_id",
            "country",
        ]
    )

    events = events.sort_values(
        ["country", "start_date", "event_id"]
    ).reset_index(drop=True)

    events.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Forecasting event catalogue created")
    print("------------------------------------")
    print("Events:", len(events))
    print("Output:", OUTPUT_FILE)
    print()

    print("Events by NEXUS country")
    print("-----------------------")

    print(
        events["country"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Date range")
    print("----------")

    print(
        "Earliest:",
        events["start_date"].min(),
    )

    print(
        "Latest:",
        events["end_date"].max(),
    )

    print()
    print("Cross-border DFO events retained:")
    print("-----------------------------------")

    cross_border = events[
        events["dfo_country"].str.contains(
            "•",
            na=False,
        )
    ]

    print("Rows:", len(cross_border))

    if len(cross_border) > 0:
        print(
            cross_border[
                [
                    "event_id",
                    "dfo_country",
                    "country",
                    "start_date",
                    "end_date",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print()
    print("Forecasting event preparation complete.")


if __name__ == "__main__":
    main()