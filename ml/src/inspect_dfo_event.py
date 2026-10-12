from pathlib import Path
import requests
from bs4 import BeautifulSoup


EVENT_ID = "4371"

URL = (
    f"https://floodobservatory.colorado.edu/"
    f"wiki/FloodReport:Report-{int(EVENT_ID):06d}"
)


print("NEXUS DFO Event Inspector")
print("=========================")
print()
print("Event:", EVENT_ID)
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

soup = BeautifulSoup(
    response.text,
    "html.parser",
)

print("HTTP status:", response.status_code)
print()

print("Links found on event page")
print("-------------------------")

links = soup.find_all("a")

for link in links:

    text = " ".join(
        link.get_text(" ", strip=True).split()
    )

    href = link.get("href")

    if not href:
        continue

    if (
        "map" in text.lower()
        or "gis" in text.lower()
        or "geotiff" in text.lower()
        or "flood" in text.lower()
        or "event" in href.lower()
        or "map" in href.lower()
        or "gis" in href.lower()
        or ".zip" in href.lower()
        or ".shp" in href.lower()
        or ".kml" in href.lower()
        or ".geojson" in href.lower()
    ):
        print(text or "[no link text]")
        print(" ", href)
        print()