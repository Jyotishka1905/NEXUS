from pathlib import Path
import json
import random

import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "forecasting_events_spatial.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_samples.csv"
)

GEOJSON_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "dfo_geojson_cache"
)

TARGET_COUNTRIES = {
    "India",
    "Ghana",
    "Sri Lanka",
}

MIN_YEAR = 2016
MAX_YEAR = 2023

MAX_EVENTS_PER_COUNTRY = 25

RANDOM_SEED = 42


def load_event_geometries():

    df = pd.read_csv(EVENT_FILE)

    required = {
        "event_id",
        "country",
        "start_date",
        "end_date",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + str(sorted(missing))
        )

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce",
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce",
    )

    df = df[
        df["country"].isin(TARGET_COUNTRIES)
        & df["start_date"].notna()
        & df["end_date"].notna()
    ].copy()

    df = df[
        (df["start_date"].dt.year >= MIN_YEAR)
        & (df["start_date"].dt.year <= MAX_YEAR)
    ].copy()

    records = []

    for event_id, group in df.groupby("event_id"):

        event_id = int(event_id)

        # Actual cache uses 000437.json, not 000437.geojson.
        geojson_path = (
            GEOJSON_DIR
            / f"{event_id:06d}.json"
        )

        if not geojson_path.exists():
            continue

        try:

            with open(
                geojson_path,
                "r",
                encoding="utf-8",
            ) as f:
                geojson = json.load(f)

            geometries = []

            for feature in geojson.get(
                "features",
                [],
            ):

                geometry = feature.get(
                    "geometry"
                )

                if geometry:
                    geometries.append(
                        shape(geometry)
                    )

            if not geometries:
                continue

            combined_geometry = unary_union(
                geometries
            )

            if combined_geometry.is_empty:
                continue

            representative_point = (
                combined_geometry.representative_point()
            )

            records.append(
                {
                    "event_id": event_id,
                    "country": group.iloc[0]["country"],
                    "start_date": group.iloc[0]["start_date"],
                    "end_date": group.iloc[0]["end_date"],
                    "latitude": representative_point.y,
                    "longitude": representative_point.x,
                    "geometry": combined_geometry,
                }
            )

        except Exception as exc:

            print(
                f"Skipping event {event_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    return records


def point_is_in_active_event(
    point,
    date,
    events,
):

    for event in events:

        if not (
            event["start_date"]
            <= date
            <= event["end_date"]
        ):
            continue

        if event["geometry"].contains(point):
            return True

    return False


def find_negative_date(
    event,
    events,
    rng,
):

    point = Point(
        event["longitude"],
        event["latitude"],
    )

    event_start = event["start_date"]
    event_end = event["end_date"]

    candidates = []

    # Search before the event.
    for offset in range(14, 121):

        candidates.append(
            event_start
            - pd.Timedelta(days=offset)
        )

    # Also search after the event.
    for offset in range(14, 121):

        candidates.append(
            event_end
            + pd.Timedelta(days=offset)
        )

    rng.shuffle(candidates)

    for candidate in candidates:

        # Candidate forecast date must not be
        # inside an active DFO event.
        if point_is_in_active_event(
            point,
            candidate,
            events,
        ):
            continue

        # D-7 through D-1 must also be free
        # of active DFO events.
        clean_history = True

        for history_offset in range(1, 8):

            history_date = (
                candidate
                - pd.Timedelta(
                    days=history_offset
                )
            )

            if point_is_in_active_event(
                point,
                history_date,
                events,
            ):

                clean_history = False
                break

        if clean_history:
            return candidate

    return None


def main():

    rng = random.Random(
        RANDOM_SEED
    )

    print()
    print(
        "NEXUS DFO Forecast Sample Builder"
    )
    print(
        "----------------------------------"
    )
    print()

    events = load_event_geometries()

    print(
        f"Usable DFO events: {len(events)}"
    )

    if not events:

        raise RuntimeError(
            "No usable DFO events found."
        )

    selected_events = []

    for country in sorted(
        TARGET_COUNTRIES
    ):

        country_events = [
            event
            for event in events
            if event["country"] == country
        ]

        country_events.sort(
            key=lambda x: x["start_date"]
        )

        if len(country_events) > MAX_EVENTS_PER_COUNTRY:

            step = (
                len(country_events)
                / MAX_EVENTS_PER_COUNTRY
            )

            chosen = []

            for i in range(
                MAX_EVENTS_PER_COUNTRY
            ):

                index = min(
                    int(i * step),
                    len(country_events) - 1,
                )

                chosen.append(
                    country_events[index]
                )

            unique = {}

            for event in chosen:
                unique[
                    event["event_id"]
                ] = event

            country_events = list(
                unique.values()
            )

        selected_events.extend(
            country_events
        )

        print(
            f"{country}: "
            f"{len(country_events)} events selected"
        )

    print()
    print(
        f"Total selected events: "
        f"{len(selected_events)}"
    )

    rows = []

    positive_count = 0
    negative_count = 0

    for event in selected_events:

        event_id = event["event_id"]

        positive_date = (
            event["start_date"]
        )

        # --------------------------------------------------------------
        # Positive sample
        # --------------------------------------------------------------

        rows.append(
            {
                "sample_id":
                    f"event_{event_id}",

                "group_id":
                    f"event_{event_id}",

                "label": 1,

                "country":
                    event["country"],

                "latitude":
                    event["latitude"],

                "longitude":
                    event["longitude"],

                "forecast_date":
                    positive_date.date(),

                "event_id":
                    event_id,

                "event_start_date":
                    event["start_date"].date(),

                "event_end_date":
                    event["end_date"].date(),

                "source":
                    "DFO_EVENT",
            }
        )

        positive_count += 1

        # --------------------------------------------------------------
        # Matched negative sample
        # --------------------------------------------------------------

        negative_date = find_negative_date(
            event,
            events,
            rng,
        )

        if negative_date is None:

            print(
                f"WARNING: no clean negative "
                f"found for event {event_id}"
            )

            # Remove the positive sample too.
            # This keeps the final dataset balanced.
            rows.pop()

            positive_count -= 1

            continue

        rows.append(
            {
                "sample_id":
                    f"control_{event_id}",

                "group_id":
                    f"event_{event_id}",

                "label": 0,

                "country":
                    event["country"],

                "latitude":
                    event["latitude"],

                "longitude":
                    event["longitude"],

                "forecast_date":
                    negative_date.date(),

                "event_id":
                    event_id,

                "event_start_date":
                    event["start_date"].date(),

                "event_end_date":
                    event["end_date"].date(),

                "source":
                    "DFO_MATCHED_CONTROL",
            }
        )

        negative_count += 1

    output = pd.DataFrame(rows)

    if output.empty:

        raise RuntimeError(
            "No forecast samples were created."
        )

    output["forecast_date"] = pd.to_datetime(
        output["forecast_date"]
    )

    output = output.sort_values(
        [
            "country",
            "forecast_date",
            "label",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Dataset created"
    )
    print(
        "---------------"
    )

    print(
        f"Positive samples: {positive_count}"
    )

    print(
        f"Negative samples: {negative_count}"
    )

    print(
        f"Total samples:    {len(output)}"
    )

    print()
    print(
        "Samples by country:"
    )

    print(
        output.groupby("country")["label"]
        .agg(["count", "sum"])
    )

    print()
    print(
        "Label distribution:"
    )

    print(
        output["label"].value_counts()
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print(
        "Completed successfully."
    )


if __name__ == "__main__":
    main()