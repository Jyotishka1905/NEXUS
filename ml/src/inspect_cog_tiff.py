import requests

product_id = "66bb87fd-d036-47eb-a86e-fe18ce0400f9"

safe_name = "S1A_IW_GRDH_1SSV_20160309T114019_20160309T114044_010292_00F39B_529A_COG.SAFE"

filename = "s1a-iw-grd-vv-20160309t114019-20160309t114044-010292-00f39b-001-cog.tiff"

url = (
    f"https://download.dataspace.copernicus.eu"
    f"/odata/v1/Products({product_id})"
    f"/Nodes({safe_name})"
    f"/Nodes(measurement)"
    f"/Nodes({filename})"
)

response = requests.get(url, timeout=30)

print("Status:", response.status_code)
print()
print(response.text[:5000])
