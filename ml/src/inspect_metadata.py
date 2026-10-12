from pathlib import Path
import json


root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
)

catalog_root = root / "catalog" / "catalog"

label_catalog = (
    catalog_root
    / "sen1floods11_hand_labeled_label"
)


json_files = sorted(label_catalog.rglob("*.json"))

print("Metadata files found:", len(json_files))
print()

countries = {}
samples = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    properties = data.get("properties", {})

    country = properties.get("country", "UNKNOWN")
    chip_id = properties.get("chip_id", "UNKNOWN")

    samples.append({
        "chip_id": chip_id,
        "country": country,
    })

    countries[country] = countries.get(country, 0) + 1


print("Countries:")
print("----------------")

for country, count in sorted(countries.items()):
    print(f"{country}: {count}")

print()
print("Total samples:", len(samples))
