from pathlib import Path
import csv
import re
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "flood_events_expanded.csv"


COUNTRIES = {
    "India": "https://floodobservatory.colorado.edu/wiki/India",
    "Sri Lanka": "https://floodobservatory.colorado.edu/wiki/Sri_Lanka",
    "Ghana": "https://floodobservatory.colorado.edu/wiki/Ghana",
}


def clean(value):
    return " ".join(value.split())


def parse_dfo_table(country, url):
    print(f"[DOWNLOAD] {country}")
    print(f"  URL: {url}")

    response = requests.get(
        url,
        timeout=(10, 20),
        headers={
            "User-Agent": "Mozilla/5.0 NEXUS research dataset builder"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    tables = soup.find_all("table")

    print(f"  Tables found: {len(tables)}")

    if not tables:
        return []

    # The DFO country pages currently contain one main event table.
    table = tables[0]

    rows = table.find_all("tr")

    records = []

    for row in rows:
        cells = row.find_all(["td", "th"])

        values = [
            clean(cell.get_text(" ", strip=True))
            for cell in cells
        ]

        if len(values) < 6:
            continue

        # Expected DFO layout:
        #
        # Event #
        # Glide #
        # Start Date
        # End Date
        # Country
        # Area
        # ...

        event_id = values[0]
        start_date = values[2]
        end_date = values[3]
        event_country = values[4]
        area = values[5]

        # Ignore header rows.
        if not re.fullmatch(r"\d+", event_id):
            continue

        # Verify date format.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_date):
            continue

        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date):
            continue

        records.append(
            {
                "event_id": event_id,
                "country": event_country,
                "start_date": start_date,
                "end_date": end_date,
                "area_km2": area.replace(",", ""),
                "target_source": "DFO",
            }
        )

    print(f"  Events extracted: {len(records)}")

    return records


def main():

    print("NEXUS DFO Flood Event Catalogue Builder")
    print("========================================")
    print()

    all_records = []

    for country, url in COUNTRIES.items():

        try:
            records = parse_dfo_table(
                country,
                url,
            )

            all_records.extend(records)

        except requests.RequestException as error:

            print()
            print(f"  ERROR downloading {country}:")
            print(f"  {error}")
            print()

        except Exception as error:

            print()
            print(f"  ERROR parsing {country}:")
            print(f"  {error}")
            print()

    if not all_records:
        raise RuntimeError(
            "No DFO events were extracted."
        )

    # Remove duplicates.
    unique_records = {}

    for record in all_records:

        key = (
            record["event_id"],
            record["country"],
            record["start_date"],
            record["end_date"],
        )

        unique_records[key] = record

    records = list(unique_records.values())

    records.sort(
        key=lambda row: (
            row["country"],
            row["start_date"],
        )
    )

    fieldnames = [
        "event_id",
        "country",
        "start_date",
        "end_date",
        "area_km2",
        "target_source",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print("DFO catalogue created")
    print("---------------------")
    print("Total events:", len(records))
    print("Output:", OUTPUT_FILE)
    print()

    print("Events by country")
    print("-----------------")

    country_counts = {}

    for record in records:
        country = record["country"]

        country_counts[country] = (
            country_counts.get(country, 0) + 1
        )

    for country, count in sorted(
        country_counts.items()
    ):
        print(f"{country}: {count}")

    print()
    print("First 15 events")
    print("----------------")

    for record in records[:15]:
        print(
            record["event_id"],
            record["country"],
            record["start_date"],
            "→",
            record["end_date"],
            "| area:",
            record["area_km2"],
        )


if __name__ == "__main__":
    main()