import requests

filter_key = chr(36) + "filter"

filter_value = (
    "contains(Name,'S1A_IW_GRDH_1SSV_20160309T114019_20160309T114044') "
    "and contains(Name,'COG')"
)

params = {
    filter_key: filter_value
}

url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

response = requests.get(url, params=params, timeout=30)

print("URL:")
print(response.url)
print()
print("Status:", response.status_code)
print()
print(response.text[:8000])
