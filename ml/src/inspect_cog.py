import requests
import json

product_id = "66bb87fd-d036-47eb-a86e-fe18ce0400f9"

url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})"

response = requests.get(url, timeout=30)

print("Status:", response.status_code)
print()

data = response.json()

print("Name:", data.get("Name"))
print("S3Path:", data.get("S3Path"))
print("ContentLength:", data.get("ContentLength"))
print()

print("Keys:")
print(list(data.keys()))
