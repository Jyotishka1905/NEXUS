import os
import requests

username = os.environ["CDSE_USERNAME"]
password = os.environ["CDSE_PASSWORD"]

url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

data = {
    "client_id": "cdse-public",
    "username": username,
    "password": password,
    "grant_type": "password",
}

response = requests.post(url, data=data, timeout=30)

print("Status:", response.status_code)
print()
print(response.text[:2000])
