import json
import re
import requests


EVENT_ID = "4371"

URL = (
    f"https://floodobservatory.colorado.edu/"
    f"wiki/GeoJson:FloodReport-{int(EVENT_ID):06d}"
)


print("NEXUS DFO Embedded GeoJSON Extractor")
print("====================================")
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

html = response.text

print("HTTP status:", response.status_code)
print("HTML length:", len(html))
print()


# Find the embedded GeoJson assignment.
matches = list(
    re.finditer(
        r"GeoJson\s*=",
        html,
        flags=re.IGNORECASE,
    )
)

print("GeoJson assignments found:", len(matches))
print()

if not matches:
    raise RuntimeError(
        "Could not find an embedded GeoJson assignment."
    )


for number, match in enumerate(matches, start=1):

    print(f"--- GeoJson assignment {number} ---")

    start = match.end()

    # Print a useful section around the assignment first.
    preview_end = min(
        len(html),
        start + 3000,
    )

    preview = html[start:preview_end]

    print(preview[:3000])
    print()


    # Try to locate a JavaScript object.
    first_brace = html.find(
        "{",
        start,
    )

    if first_brace == -1:
        print("No opening JSON brace found.")
        continue


    # Balanced-brace extraction.
    depth = 0
    in_string = False
    escape = False
    end = None

    for index in range(
        first_brace,
        len(html),
    ):

        char = html[index]

        if in_string:

            if escape:
                escape = False

            elif char == "\\":
                escape = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                end = index + 1
                break


    if end is None:
        print("Could not find balanced JSON object.")
        continue


    candidate = html[
        first_brace:end
    ].strip()

    print("Extracted object length:", len(candidate))
    print()

    try:

        data = json.loads(candidate)

        print("VALID JSON FOUND")
        print("================")

        print(
            json.dumps(
                data,
                indent=2,
            )[:10000]
        )

    except json.JSONDecodeError as error:

        print("Object is not directly valid JSON.")
        print("JSON error:", error)
        print()
        print("First 5000 characters:")
        print(candidate[:5000])

    print()