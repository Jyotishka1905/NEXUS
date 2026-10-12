from pathlib import Path
import json
import re
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "forecasting_events.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "forecasting_events_spatial.csv"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "dfo_geojson_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 NEXUS research project"
    }
)


def extract_geojson(event_id):
    """
    Download and extract the embedded GeoJSON for a DFO event.

    Uses a local cache so successfully downloaded geometries
    do not need to be requested again.
    """

    event_id = int(event_id)

    url = (
        "https://floodobservatory.colorado.edu/"
        f"wiki/GeoJson:FloodReport-{event_id:06d}"
    )

    cache_file = (
        CACHE_DIR
        / f"{event_id:06d}.json"
    )

    # ---------------------------------------------------------
    # Use cached geometry if available.
    # ---------------------------------------------------------

    if cache_file.exists():

        try:
            with open(
                cache_file,
                "r",
                encoding="utf-8",
            ) as file:

                geojson = json.load(file)

            return geojson

        except (
            json.JSONDecodeError,
            OSError,
        ):

            print(
                f"[CACHE ERROR] {event_id}: "
                "cached file could not be read"
            )

            # Continue and attempt a fresh download.


    # ---------------------------------------------------------
    # Download DFO GeoJSON page.
    # ---------------------------------------------------------

    response = SESSION.get(
        url,
        timeout=(10, 30),
    )

    if response.status_code != 200:

        print(
            f"[SKIP] {event_id}: "
            f"HTTP {response.status_code}"
        )

        return None


    html = response.text


    # ---------------------------------------------------------
    # Find embedded:
    #
    # GeoJson = { ... };
    # ---------------------------------------------------------

    match = re.search(
        r"GeoJson\s*=",
        html,
        flags=re.IGNORECASE,
    )

    if match is None:

        print(
            f"[SKIP] {event_id}: "
            "GeoJson assignment not found"
        )

        return None


    start = match.end()


    first_brace = html.find(
        "{",
        start,
    )

    if first_brace == -1:

        print(
            f"[SKIP] {event_id}: "
            "opening JSON object not found"
        )

        return None


    # ---------------------------------------------------------
    # Extract balanced JSON object.
    #
    # This handles nested objects and arrays and avoids
    # accidentally stopping at an inner closing brace.
    # ---------------------------------------------------------

    depth = 0
    in_string = False
    escape = False
    end = None


    for index in range(
        first_brace,
        len(html),
    ):

        char = html[index]


        if in_string:

            if escape:

                escape = False

            elif char == "\\":

                escape = True

            elif char == '"':

                in_string = False

            continue


        if char == '"':

            in_string = True

        elif char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                end = index + 1

                break


    if end is None:

        print(
            f"[SKIP] {event_id}: "
            "could not find complete JSON object"
        )

        return None


    candidate = html[
        first_brace:end
    ]


    # ---------------------------------------------------------
    # Parse JSON.
    # ---------------------------------------------------------

    try:

        geojson = json.loads(
            candidate
        )

    except json.JSONDecodeError as error:

        print(
            f"[SKIP] {event_id}: "
            f"invalid GeoJSON: {error}"
        )

        return None


    # ---------------------------------------------------------
    # Basic GeoJSON validation.
    # ---------------------------------------------------------

    if geojson.get("type") != "FeatureCollection":

        print(
            f"[SKIP] {event_id}: "
            "not a FeatureCollection"
        )

        return None


    features = geojson.get(
        "features",
        [],
    )

    if not features:

        print(
            f"[SKIP] {event_id}: "
            "no features"
        )

        return None


    # ---------------------------------------------------------
    # Cache successful GeoJSON.
    # ---------------------------------------------------------

    try:

        with open(
            cache_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                geojson,
                file,
            )

    except OSError as error:

        print(
            f"[CACHE WARNING] {event_id}: "
            f"could not save cache: {error}"
        )


    return geojson


def geometry_bounds(geojson):
    """
    Calculate longitude/latitude bounds from all geometry
    coordinates contained in a GeoJSON FeatureCollection.
    """

    coordinates = []


    for feature in geojson.get(
        "features",
        [],
    ):

        geometry = feature.get(
            "geometry"
        )

        if not geometry:

            continue


        geometry_coordinates = geometry.get(
            "coordinates"
        )


        def collect_coords(value):

            # A GeoJSON coordinate pair is:
            # [longitude, latitude]

            if (
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(item, (int, float))
                    for item in value
                )
            ):

                coordinates.append(
                    (
                        float(value[0]),
                        float(value[1]),
                    )
                )

                return


            if isinstance(value, list):

                for item in value:

                    collect_coords(item)


        collect_coords(
            geometry_coordinates
        )


    if not coordinates:

        return None


    longitudes = [
        coordinate[0]
        for coordinate in coordinates
    ]

    latitudes = [
        coordinate[1]
        for coordinate in coordinates
    ]


    return {
        "min_longitude": min(longitudes),
        "min_latitude": min(latitudes),
        "max_longitude": max(longitudes),
        "max_latitude": max(latitudes),
    }


def main():

    print(
        "NEXUS DFO Spatial Catalogue Builder"
    )

    print(
        "===================================="
    )

    print()


    # ---------------------------------------------------------
    # Load forecasting event associations.
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )


    events = pd.read_csv(
        INPUT_FILE
    )


    required_columns = {
        "event_id",
        "country",
        "start_date",
        "end_date",
    }


    missing_columns = (
        required_columns
        - set(events.columns)
    )


    if missing_columns:

        raise RuntimeError(
            "Input file is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


    print(
        "Input event associations:",
        len(events),
    )

    print()


    records = []

    success = 0
    failed = 0


    # ---------------------------------------------------------
    # Process every DFO event association.
    # ---------------------------------------------------------

    for index, row in events.iterrows():

        event_id = str(
            row["event_id"]
        )


        print(
            f"[{index + 1}/{len(events)}] "
            f"Event {event_id}"
        )


        try:

            geojson = extract_geojson(
                event_id
            )

        except requests.RequestException as error:

            print(
                f"  Request failed: {error}"
            )

            geojson = None


        if geojson is None:

            failed += 1

            continue


        # -----------------------------------------------------
        # Calculate spatial bounds.
        # -----------------------------------------------------

        bounds = geometry_bounds(
            geojson
        )


        if bounds is None:

            print(
                "  No geometry coordinates found."
            )

            failed += 1

            continue


        # -----------------------------------------------------
        # Add spatial information to original event row.
        # -----------------------------------------------------

        record = row.to_dict()

        record.update(
            bounds
        )

        record["geometry_features"] = len(
            geojson.get(
                "features",
                [],
            )
        )


        records.append(
            record
        )

        success += 1


        print(
            "  Geometry:",
            record["geometry_features"],
            "feature(s)",
        )


        # IMPORTANT:
        # Use ASCII "->" instead of Unicode "->"
        # so Windows PowerShell cp1252 does not crash.

        print(
            "  Bounds:",
            f"{bounds['min_longitude']:.4f},",
            f"{bounds['min_latitude']:.4f} ->",
            f"{bounds['max_longitude']:.4f},",
            f"{bounds['max_latitude']:.4f}",
        )


        # -----------------------------------------------------
        # Be polite to the public DFO server.
        # -----------------------------------------------------

        time.sleep(0.2)


    # ---------------------------------------------------------
    # Build output dataframe.
    # ---------------------------------------------------------

    spatial_df = pd.DataFrame(
        records
    )


    if spatial_df.empty:

        raise RuntimeError(
            "No spatial DFO events were extracted."
        )


    # ---------------------------------------------------------
    # Save spatial catalogue.
    # ---------------------------------------------------------

    spatial_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    # ---------------------------------------------------------
    # Final summary.
    # ---------------------------------------------------------

    print()

    print(
        "DFO Spatial Catalogue"
    )

    print(
        "---------------------"
    )

    print(
        "Successful:",
        success,
    )

    print(
        "Failed:",
        failed,
    )

    print(
        "Total output rows:",
        len(spatial_df),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print()


    print(
        "Rows by country:"
    )


    print(
        spatial_df[
            "country"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print()

    print(
        "Spatial catalogue completed successfully."
    )


if __name__ == "__main__":

    main()