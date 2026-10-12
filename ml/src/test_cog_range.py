import os
import requests

AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

PRODUCT_ID = "66bb87fd-d036-47eb-a86e-fe18ce0400f9"

SAFE = "S1A_IW_GRDH_1SSV_20160309T114019_20160309T114044_010292_00F39B_529A_COG.SAFE"

FILE = "s1a-iw-grd-vv-20160309t114019-20160309t114044-010292-00f39b-001-cog.tiff"

# Get token
auth_data = {
    "client_id": "cdse-public",
    "grant_type": "password",
    "username": os.environ["CDSE_USERNAME"],
    "password": os.environ["CDSE_PASSWORD"],
}

auth = requests.post(AUTH_URL, data=auth_data, timeout=30)
auth.raise_for_status()

token = auth.json()["access_token"]

# Individual file endpoint
url = (
    f"https://download.dataspace.copernicus.eu"
    f"/odata/v1/Products({PRODUCT_ID})"
    f"/Nodes({SAFE})"
    f"/Nodes(measurement)"
    f"/Nodes({FILE})"
    f"/"
)

headers = {
    "Authorization": f"Bearer {token}",
    "Range": "bytes=0-1023",
}

response = requests.get(
    url,
    headers=headers,
    timeout=60,
    allow_redirects=True,
)

print("HTTP status:", response.status_code)
print("Content-Length:", response.headers.get("Content-Length"))
print("Content-Range:", response.headers.get("Content-Range"))
print("Content-Type:", response.headers.get("Content-Type"))
print("Bytes received:", len(response.content))
print("Response body:", response.text)