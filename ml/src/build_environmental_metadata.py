from pathlib import Path
import json
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "catalog"
    / "catalog"
    / "sen1floods11_hand_labeled_source"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "environmental" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "samples.csv"


def extract_metadata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    properties = data.get("properties", {})
    bbox = data.get("bbox")

    if not bbox or len(bbox) != 4:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    return {
        "chip_id": properties.get("chip_id"),
        "country": properties.get("country"),
        "latitude": center_lat,
        "longitude": center_lon,
        "datetime": properties.get("datetime"),
        "s1_date": properties.get("datetime", "")[:10],
    }


def main():
    json_files = sorted(CATALOG_ROOT.rglob("*.json"))

    print("NEXUS Environmental Metadata")
    print("============================")
    print("Catalog:", CATALOG_ROOT)
    print("JSON files found:", len(json_files))
    print()

    records = []

    for json_path in json_files:
        record = extract_metadata(json_path)

        if record is not None and record["chip_id"] is not None:
            records.append(record)

    fieldnames = [
        "chip_id",
        "country",
        "latitude",
        "longitude",
        "datetime",
        "s1_date",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("Records written:", len(records))
    print("Saved to:", OUTPUT_FILE)
    print()

    print("First 5 records:")
    print("----------------")

    for record in records[:5]:
        print(record)


if __name__ == "__main__":
    main()