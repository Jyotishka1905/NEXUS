from pathlib import Path
import json

import pandas as pd
from shapely.geometry import Point, shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "forecasting_events.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "dfo_geojson_cache"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "sen1_dfo_matches.csv"
)


# ============================================================
# LOAD DFO GEOJSON
# ============================================================

def load_geojson(event_id):

    event_id = int(event_id)

    cache_file = (
        CACHE_DIR
        / f"{event_id:06d}.json"
    )

    if not cache_file.exists():
        return None

    try:
        with open(
            cache_file,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None


# ============================================================
# BUILD DFO GEOMETRIES
# ============================================================

def build_event_geometries(events):

    """
    Build one geometry record for each unique DFO event.

    A DFO event can have multiple country associations.
    The geometry itself belongs to the event, so duplicate
    event IDs are combined.
    """

    grouped_events = (
        events
        .groupby("event_id", as_index=False)
        .agg(
            {
                "country": lambda values: " | ".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
                "start_date": "min",
                "end_date": "max",
                "area_km2": "first",
            }
        )
    )

    print(
        "Loading DFO event geometries:",
        len(grouped_events),
    )

    print()

    event_geometries = []

    missing_geometry = 0
    invalid_geometry = 0

    for _, row in grouped_events.iterrows():

        event_id = row["event_id"]

        geojson = load_geojson(
            event_id
        )

        if geojson is None:

            missing_geometry += 1

            continue

        geometries = []

        for feature in geojson.get(
            "features",
            [],
        ):

            geometry_data = feature.get(
                "geometry"
            )

            if not geometry_data:
                continue

            try:

                geometry = shape(
                    geometry_data
                )

                if geometry.is_empty:
                    continue

                if not geometry.is_valid:

                    geometry = geometry.buffer(
                        0
                    )

                if not geometry.is_empty:

                    geometries.append(
                        geometry
                    )

            except Exception:

                invalid_geometry += 1

        if not geometries:
            continue

        event_geometries.append(
            {
                "event_id": event_id,
                "country": row["country"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "area_km2": row["area_km2"],
                "geometries": geometries,
            }
        )

    print(
        "Loaded DFO events with geometry:",
        len(event_geometries),
    )

    print(
        "Events without usable geometry:",
        missing_geometry,
    )

    print(
        "Invalid geometry records skipped:",
        invalid_geometry,
    )

    print()

    return event_geometries


# ============================================================
# POINT + EVENT MATCH
# ============================================================

def point_matches_event(
    point,
    observation_date,
    event,
):

    """
    A sample matches an event when:

    1. Its observation date falls inside the DFO event
       start/end interval.

    2. Its geographic center intersects the DFO geometry.
    """

    event_start = pd.Timestamp(
        event["start_date"]
    )

    event_end = pd.Timestamp(
        event["end_date"]
    )

    # --------------------------------------------------------
    # Temporal condition
    # --------------------------------------------------------

    if (
        observation_date < event_start
        or observation_date > event_end
    ):

        return False

    # --------------------------------------------------------
    # Spatial condition
    # --------------------------------------------------------

    for geometry in event["geometries"]:

        if geometry.intersects(
            point
        ):

            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "NEXUS Sen1Floods11 <-> DFO Event Matcher"
    )

    print(
        "=========================================="
    )

    print()

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not SAMPLES_FILE.exists():

        raise FileNotFoundError(
            f"Samples file not found: {SAMPLES_FILE}"
        )

    if not EVENTS_FILE.exists():

        raise FileNotFoundError(
            f"Events file not found: {EVENTS_FILE}"
        )

    if not CACHE_DIR.exists():

        raise FileNotFoundError(
            f"DFO GeoJSON cache not found: {CACHE_DIR}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    samples = pd.read_csv(
        SAMPLES_FILE
    )

    events = pd.read_csv(
        EVENTS_FILE
    )

    print(
        "Sen1 samples:",
        len(samples),
    )

    print(
        "DFO event-country rows:",
        len(events),
    )

    print()

    # ========================================================
    # CHECK REQUIRED SAMPLE COLUMNS
    # ========================================================

    required_sample_columns = {
        "chip_id",
        "country",
        "latitude",
        "longitude",
    }

    missing_sample_columns = (
        required_sample_columns
        - set(samples.columns)
    )

    if missing_sample_columns:

        raise RuntimeError(
            "samples.csv is missing required columns: "
            + ", ".join(
                sorted(
                    missing_sample_columns
                )
            )
        )

    # ========================================================
    # DETERMINE OBSERVATION DATE
    # ========================================================

    # Your actual samples.csv contains:
    #
    # datetime
    # s1_date
    #
    # Use s1_date first because it directly represents
    # the Sentinel-1 observation date.

    if "s1_date" in samples.columns:

        sample_date_column = "s1_date"

    elif "datetime" in samples.columns:

        sample_date_column = "datetime"

    else:

        raise RuntimeError(
            "samples.csv must contain either "
            "'s1_date' or 'datetime'."
        )

    print(
        "Using Sen1 observation date column:",
        sample_date_column,
    )

    print()

    # ========================================================
    # TARGET COUNTRIES
    # ========================================================

    target_countries = {
        "India",
        "Sri Lanka",
        "Ghana",
    }

    samples = samples[
        samples["country"].isin(
            target_countries
        )
    ].copy()

    events = events[
        events["country"].isin(
            target_countries
        )
    ].copy()

    print(
        "Sen1 samples in DFO countries:",
        len(samples),
    )

    print(
        "DFO events in target countries:",
        len(events),
    )

    print()

    # ========================================================
    # PARSE SAMPLE DATES
    # ========================================================

    samples["observation_date"] = pd.to_datetime(
        samples[
            sample_date_column
        ],
        errors="coerce",
    )

    # ========================================================
    # PARSE EVENT DATES
    # ========================================================

    events["start_date"] = pd.to_datetime(
        events["start_date"],
        errors="coerce",
    )

    events["end_date"] = pd.to_datetime(
        events["end_date"],
        errors="coerce",
    )

    # ========================================================
    # PARSE COORDINATES
    # ========================================================

    samples["latitude"] = pd.to_numeric(
        samples["latitude"],
        errors="coerce",
    )

    samples["longitude"] = pd.to_numeric(
        samples["longitude"],
        errors="coerce",
    )

    # ========================================================
    # INVALID SAMPLE CHECK
    # ========================================================

    invalid_sample_dates = samples[
        samples["observation_date"].isna()
    ]

    invalid_coordinates = samples[
        samples["latitude"].isna()
        |
        samples["longitude"].isna()
        |
        (samples["latitude"] < -90)
        |
        (samples["latitude"] > 90)
        |
        (samples["longitude"] < -180)
        |
        (samples["longitude"] > 180)
    ]

    print(
        "Samples with invalid dates:",
        len(invalid_sample_dates),
    )

    print(
        "Samples with invalid coordinates:",
        len(invalid_coordinates),
    )

    print()

    # ========================================================
    # INVALID EVENT CHECK
    # ========================================================

    invalid_event_dates = events[
        events["start_date"].isna()
        |
        events["end_date"].isna()
    ]

    print(
        "Events with invalid dates:",
        len(invalid_event_dates),
    )

    print()

    # ========================================================
    # REMOVE INVALID SAMPLE RECORDS
    # ========================================================

    samples = samples[
        samples["observation_date"].notna()
        &
        samples["latitude"].notna()
        &
        samples["longitude"].notna()
        &
        (samples["latitude"] >= -90)
        &
        (samples["latitude"] <= 90)
        &
        (samples["longitude"] >= -180)
        &
        (samples["longitude"] <= 180)
    ].copy()

    # ========================================================
    # REMOVE INVALID EVENTS
    # ========================================================

    events = events[
        events["start_date"].notna()
        &
        events["end_date"].notna()
    ].copy()

    # ========================================================
    # LOAD GEOMETRIES
    # ========================================================

    event_geometries = build_event_geometries(
        events
    )

    if not event_geometries:

        raise RuntimeError(
            "No DFO geometries could be loaded."
        )

    # ========================================================
    # MATCH SAMPLES
    # ========================================================

    results = []

    for _, row in samples.iterrows():

        chip_id = row[
            "chip_id"
        ]

        country = row[
            "country"
        ]

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        observation_date = pd.Timestamp(
            row["observation_date"]
        )

        point = Point(
            longitude,
            latitude,
        )

        matches = []

        # ----------------------------------------------------
        # Test against every DFO event
        # ----------------------------------------------------

        for event in event_geometries:

            if point_matches_event(
                point,
                observation_date,
                event,
            ):

                matches.append(
                    event
                )

        # ====================================================
        # MATCH FOUND
        # ====================================================

        if matches:

            event = matches[0]

            forecast_target = 1

            matched_event_id = event[
                "event_id"
            ]

            matched_event_country = event[
                "country"
            ]

            event_start_date = (
                pd.Timestamp(
                    event["start_date"]
                ).strftime(
                    "%Y-%m-%d"
                )
            )

            event_end_date = (
                pd.Timestamp(
                    event["end_date"]
                ).strftime(
                    "%Y-%m-%d"
                )
            )

            event_area_km2 = event[
                "area_km2"
            ]

        # ====================================================
        # NO MATCH
        # ====================================================

        else:

            forecast_target = 0

            matched_event_id = ""

            matched_event_country = ""

            event_start_date = ""

            event_end_date = ""

            event_area_km2 = ""

        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append(
            {
                "chip_id": chip_id,
                "country": country,
                "latitude": latitude,
                "longitude": longitude,
                "observation_date": (
                    observation_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "matched_event_id": (
                    matched_event_id
                ),
                "matched_event_country": (
                    matched_event_country
                ),
                "event_start_date": (
                    event_start_date
                ),
                "event_end_date": (
                    event_end_date
                ),
                "event_area_km2": (
                    event_area_km2
                ),
                "matching_event_count": len(
                    matches
                ),
                "forecast_target": (
                    forecast_target
                ),
            }
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:

        raise RuntimeError(
            "No matching records were produced."
        )

    # ========================================================
    # SAVE
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        result_df
    )

    positive = int(
        (
            result_df[
                "forecast_target"
            ]
            == 1
        ).sum()
    )

    negative = int(
        (
            result_df[
                "forecast_target"
            ]
            == 0
        ).sum()
    )

    print()

    print(
        "Spatial + temporal matching complete"
    )

    print(
        "------------------------------------"
    )

    print(
        "Total samples checked:",
        total,
    )

    print(
        "DFO-matched positives:",
        positive,
    )

    print(
        "No DFO event matched:",
        negative,
    )

    print(
        "Positive percentage:",
        (
            f"{positive / total * 100:.2f}%"
            if total
            else "0.00%"
        ),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print()

    # ========================================================
    # COUNTRY SUMMARY
    # ========================================================

    print(
        "Samples checked by country:"
    )

    print(
        result_df[
            "country"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Positive matches by country:"
    )

    positive_by_country = result_df[
        result_df[
            "forecast_target"
        ]
        == 1
    ]

    if positive_by_country.empty:

        print(
            "No positive matches."
        )

    else:

        print(
            positive_by_country[
                "country"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print()

    # ========================================================
    # EXAMPLES
    # ========================================================

    print(
        "Matched event examples:"
    )

    matched_examples = result_df[
        result_df[
            "forecast_target"
        ]
        == 1
    ].head(10)

    if matched_examples.empty:

        print(
            "No DFO spatial/date matches found."
        )

    else:

        print(
            matched_examples[
                [
                    "chip_id",
                    "country",
                    "observation_date",
                    "matched_event_id",
                    "matched_event_country",
                    "matching_event_count",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "Matching finished successfully."
    )


if __name__ == "__main__":

    main()