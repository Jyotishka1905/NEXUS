from pathlib import Path

import rasterio
import numpy as np
import torch

from preprocessing import preprocess_sentinel1


root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)

s1_dir = next(root.rglob("S1Hand"))
sample = sorted(s1_dir.glob("*.tif"))[0]

with rasterio.open(sample) as src:
    image = src.read().astype(np.float32)

processed, valid_mask = preprocess_sentinel1(image)

print("Sample:", sample.name)
print("Original shape:", image.shape)
print("Processed shape:", processed.shape)
print("Processed dtype:", processed.dtype)
print("Processed min:", processed.min().item())
print("Processed max:", processed.max().item())
print("Valid pixels:", valid_mask.sum().item())
print("Invalid pixels:", (~valid_mask).sum().item())
print("Contains NaN:", torch.isnan(processed).any().item())
