import os
import requests

AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PRODUCT_ID = "66bb87fd-d036-47eb-a86e-fe18ce0400f9"

DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu"
CATALOGUE_BASE = "https://catalogue.dataspace.copernicus.eu"

auth_data = {
    "client_id": "cdse-public",
    "grant_type": "password",
    "username": os.environ["CDSE_USERNAME"],
    "password": os.environ["CDSE_PASSWORD"],
}

# ---------------------------------------------------------
# Authenticate
# ---------------------------------------------------------

auth = requests.post(
    AUTH_URL,
    data=auth_data,
    timeout=30,
)

auth.raise_for_status()

token = auth.json()["access_token"]

headers = {
    "Authorization": f"Bearer {token}",
    "Range": "bytes=0-1023",
}

# ---------------------------------------------------------
# Get product root
# ---------------------------------------------------------

root_url = (
    f"{DOWNLOAD_BASE}/odata/v1/"
    f"Products({PRODUCT_ID})/Nodes"
)

response = requests.get(
    root_url,
    headers={"Authorization": f"Bearer {token}"},
    timeout=60,
)

response.raise_for_status()

root = response.json()["result"]

safe = root[0]

safe_nodes_url = safe["Nodes"]["uri"]

print("SAFE:")
print(safe["Name"])
print()

# ---------------------------------------------------------
# Get SAFE children
# ---------------------------------------------------------

response = requests.get(
    safe_nodes_url,
    headers={"Authorization": f"Bearer {token}"},
    timeout=60,
)

response.raise_for_status()

children = response.json()["result"]

measurement = next(
    child for child in children
    if child["Name"] == "measurement"
)

measurement_url = measurement["Nodes"]["uri"]

# ---------------------------------------------------------
# Get measurement file
# ---------------------------------------------------------

response = requests.get(
    measurement_url,
    headers={"Authorization": f"Bearer {token}"},
    timeout=60,
)

response.raise_for_status()

measurement_children = response.json()["result"]

tiff = measurement_children[0]

print("TIFF:")
print(tiff["Name"])
print("Size:", tiff["ContentLength"])
print()

# ---------------------------------------------------------
# Construct download URL from the exact returned nodes
# ---------------------------------------------------------

file_name = tiff["Name"]

download_url = (
    measurement_url
    + "("
    + file_name
    + ")"
    + "/$value"
)

print("DOWNLOAD URL:")
print(download_url)
print()

# ---------------------------------------------------------
# Test download endpoint
# ---------------------------------------------------------

response = requests.get(
    download_url,
    headers=headers,
    timeout=60,
    allow_redirects=True,
)

print("STATUS:", response.status_code)
print("CONTENT TYPE:", response.headers.get("Content-Type"))
print("CONTENT LENGTH:", response.headers.get("Content-Length"))
print("CONTENT RANGE:", response.headers.get("Content-Range"))
print("BYTES RECEIVED:", len(response.content))

if response.status_code >= 400:
    print()
    print("ERROR:")
    print(response.text)