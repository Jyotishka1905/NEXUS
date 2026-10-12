from pathlib import Path
import json
import csv


project_root = Path(__file__).resolve().parents[1]

data_root = (
    project_root
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)

catalog_root = (
    project_root
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "catalog"
    / "catalog"
    / "sen1floods11_hand_labeled_label"
)

s1_dir = next(data_root.rglob("S1Hand"))
label_dir = next(data_root.rglob("LabelHand"))

output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


# Countries are assigned as complete groups.
# This prevents chips from the same country from
# appearing in both training and evaluation sets.

split_by_country = {
    "train": {
        "Bolivia",
        "Ghana",
        "India",
        "Mekong",
        "Nigeria",
        "Paraguay",
        "Somalia",
        "Sri-Lanka",
    },
    "val": {
        "Pakistan",
        "Spain",
    },
    "test": {
        "USA",
    },
}


samples = []

for metadata_path in sorted(catalog_root.rglob("*.json")):

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    properties = metadata.get("properties", {})

    country = properties.get("country", "UNKNOWN")
    chip_id = properties.get("chip_id")

    if chip_id is None:
        continue

    # Construct the expected local filenames.
    stem = f"{country}_{chip_id}"

    image_path = s1_dir / f"{stem}_S1Hand.tif"
    label_path = label_dir / f"{stem}_LabelHand.tif"

    # Ignore metadata entries for which the actual
    # local image/label pair is missing.
    if not image_path.exists() or not label_path.exists():
        continue

    split = None

    for split_name, countries in split_by_country.items():
        if country in countries:
            split = split_name
            break

    if split is None:
        print(f"WARNING: {country} is not assigned to a split: {stem}")
        continue

    samples.append({
        "chip_id": chip_id,
        "country": country,
        "datetime": properties.get("datetime", ""),
        "image": str(image_path),
        "label": str(label_path),
        "split": split,
    })


# Write one CSV containing the complete split definition.
csv_path = output_dir / "split.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "chip_id",
            "country",
            "datetime",
            "image",
            "label",
            "split",
        ],
    )

    writer.writeheader()
    writer.writerows(samples)


print("Split created successfully.")
print()
print("Total usable samples:", len(samples))
print()

for split_name in ["train", "val", "test"]:

    subset = [
        sample
        for sample in samples
        if sample["split"] == split_name
    ]

    print(f"{split_name.upper()}: {len(subset)}")

    country_counts = {}

    for sample in subset:
        country = sample["country"]
        country_counts[country] = country_counts.get(country, 0) + 1

    for country, count in sorted(country_counts.items()):
        print(f"  {country}: {count}")

print()
print("Saved to:")
print(csv_path)
