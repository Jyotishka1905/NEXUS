import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


EVENT_ID = "4371"

URL = (
    "https://floodobservatory.colorado.edu/"
    "Events/2016India4371/2016India4371.html"
)


print("NEXUS DFO GIS Inspector")
print("========================")
print()
print("URL:", URL)
print()

response = requests.get(
    URL,
    timeout=(10, 60),
    headers={
        "User-Agent": "Mozilla/5.0 NEXUS research project"
    },
)

print("HTTP status:", response.status_code)
print()

response.raise_for_status()

soup = BeautifulSoup(
    response.text,
    "html.parser",
)

print("Relevant links")
print("--------------")

for link in soup.find_all("a"):

    text = " ".join(
        link.get_text(" ", strip=True).split()
    )

    href = link.get("href")

    if not href:
        continue

    full_url = urljoin(URL, href)

    lower = (
        text + " " + href
    ).lower()

    keywords = [
        "gis",
        "shp",
        "mapinfo",
        "map sheet",
        "automated",
        "flood map",
        "download",
        ".zip",
        ".shp",
        ".kml",
        ".geojson",
    ]

    if any(keyword in lower for keyword in keywords):

        print(text or "[no text]")
        print(" ", full_url)
        print()


print()
print("All external/file-like links")
print("-----------------------------")

for link in soup.find_all("a"):

    href = link.get("href")

    if not href:
        continue

    full_url = urljoin(URL, href)

    if (
        "floodobservatory" in full_url.lower()
        or any(
            extension in full_url.lower()
            for extension in [
                ".shp",
                ".zip",
                ".kml",
                ".geojson",
                ".tar",
                ".gz",
            ]
        )
    ):
        text = " ".join(
            link.get_text(" ", strip=True).split()
        )

        print(
            text or "[no text]",
            "->",
            full_url,
        )