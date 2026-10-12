from pathlib import Path
import rasterio
import numpy as np

root = Path(r".\data\raw\sen1floods11\v1.2\data")
s1_dir = next(root.rglob("S1Hand"))
files = list(s1_dir.glob("*.tif"))

nan_total = 0
pixel_total = 0

vv_min = np.inf
vv_max = -np.inf
vh_min = np.inf
vh_max = -np.inf

print("Files:", len(files))

for f in files:
    with rasterio.open(f) as src:
        x = src.read().astype(np.float32)

    nan_total += np.isnan(x).sum()
    pixel_total += x.size

    vv = x[0][~np.isnan(x[0])]
    vh = x[1][~np.isnan(x[1])]

    if vv.size:
        vv_min = min(vv_min, float(vv.min()))
        vv_max = max(vv_max, float(vv.max()))

    if vh.size:
        vh_min = min(vh_min, float(vh.min()))
        vh_max = max(vh_max, float(vh.max()))

print("NaN percentage:", 100 * nan_total / pixel_total)
print("VV global min/max:", vv_min, vv_max)
print("VH global min/max:", vh_min, vh_max)
