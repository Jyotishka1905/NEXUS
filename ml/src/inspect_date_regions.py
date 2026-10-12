import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "metadata"
    / "samples.csv"
)


groups = defaultdict(list)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        groups[row["s1_date"]].append(row)


print("NEXUS Geographic Distribution")
print("=============================")
print()

print(
    f"{'Date':<15}"
    f"{'Samples':>8}"
    f"{'Lat range':>25}"
    f"{'Lon range':>27}"
)

print("-" * 75)


for date, samples in sorted(groups.items()):

    latitudes = [
        float(sample["latitude"])
        for sample in samples
    ]

    longitudes = [
        float(sample["longitude"])
        for sample in samples
    ]

    print(
        f"{date:<15}"
        f"{len(samples):>8}"
        f"{min(latitudes):>10.2f} to {max(latitudes):>10.2f}"
        f"{min(longitudes):>12.2f} to {max(longitudes):>10.2f}"
    )