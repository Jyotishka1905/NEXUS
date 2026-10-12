from pathlib import Path
import json


root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
)

catalog = (
    root
    / "catalog"
    / "catalog"
    / "sen1floods11_hand_labeled_label"
)

files = sorted(catalog.rglob("*.json"))

print("Total metadata files:", len(files))
print()

for path in files[:3]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 70)
    print("FILE:", path.name)
    print()
    print("Properties:")
    
    for key, value in data.get("properties", {}).items():
        print(f"  {key}: {value}")

    print()
