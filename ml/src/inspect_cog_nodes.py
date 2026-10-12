import requests

product_id = "66bb87fd-d036-47eb-a86e-fe18ce0400f9"

url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Nodes"

response = requests.get(url, timeout=30)

print("URL:", url)
print("Status:", response.status_code)
print()
print(response.text[:8000])
