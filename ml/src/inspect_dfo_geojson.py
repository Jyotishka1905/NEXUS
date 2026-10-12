import requests
import re


EVENT_ID = "4371"

URL = (
    f"https://floodobservatory.colorado.edu/"
    f"wiki/GeoJson:FloodReport-{int(EVENT_ID):06d}"
)


print("NEXUS DFO GeoJSON Endpoint Inspector")
print("====================================")
print()
print("URL:", URL)
print()

response = requests.get(
    URL,
    timeout=(10, 30),
    headers={
        "User-Agent": "Mozilla/5.0 NEXUS research project"
    },
)

response.raise_for_status()

html = response.text

print("HTTP status:", response.status_code)
print("HTML length:", len(html))
print()

# Search for anything that looks like a GeoJSON,
# JSON, API, map, or event-data endpoint.
patterns = [
    r'https?://[^"\']+',
    r'[^"\']+\.geojson[^"\']*',
    r'[^"\']+\.json[^"\']*',
    r'[^"\']+\.js[^"\']*',
    r'api[^"\']*',
    r'GeoJson[^"\']*',
    r'FloodReport[^"\']*',
]

found = set()

for pattern in patterns:

    matches = re.findall(
        pattern,
        html,
        flags=re.IGNORECASE,
    )

    for match in matches:

        match = match.strip()

        if len(match) > 500:
            continue

        found.add(match)


print("Potential endpoints / references")
print("---------------------------------")

for item in sorted(found):

    lower = item.lower()

    if (
        "geojson" in lower
        or ".json" in lower
        or "api" in lower
        or "floodreport" in lower
        or "map" in lower
        or ".js" in lower
    ):
        print(item)