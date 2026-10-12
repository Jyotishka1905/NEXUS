from pathlib import Path
import json


root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
)

label_catalog = (
    root
    / "catalog"
    / "catalog"
    / "sen1floods11_hand_labeled_label"
)

json_files = sorted(label_catalog.rglob("*.json"))

events = {}

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    properties = data.get("properties", {})

    country = properties.get("country", "UNKNOWN")
    event = properties.get("event", None)

    if event is None:
        # Some catalog versions may encode the event in the ID/path.
        event = properties.get("event_name", None)

    if event is None:
        event = "UNKNOWN"

    key = (country, event)
    events[key] = events.get(key, 0) + 1


print("Country / event groups:")
print("------------------------")

for (country, event), count in sorted(events.items()):
    print(f"{country:12} | {str(event):30} | {count}")

print()
print("Total groups:", len(events))
print("Total samples:", sum(events.values()))
