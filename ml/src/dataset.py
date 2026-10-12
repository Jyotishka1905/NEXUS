from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_sentinel1, preprocess_label


class Sen1Floods11Dataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)

        s1_matches = list(self.data_root.rglob("S1Hand"))
        label_matches = list(self.data_root.rglob("LabelHand"))

        if not s1_matches:
            raise RuntimeError(
                f"Could not find S1Hand anywhere inside {self.data_root}"
            )

        if not label_matches:
            raise RuntimeError(
                f"Could not find LabelHand anywhere inside {self.data_root}"
            )

        self.s1_dir = s1_matches[0]
        self.label_dir = label_matches[0]

        self.images = sorted(self.s1_dir.glob("*.tif"))

        self.samples = []

        for image_path in self.images:
            label_name = image_path.name.replace(
                "_S1Hand.tif",
                "_LabelHand.tif"
            )

            label_path = self.label_dir / label_name

            if label_path.exists():
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(
                "No matching Sentinel-1 image / label pairs found."
            )

        print(f"S1 directory:    {self.s1_dir}")
        print(f"Label directory: {self.label_dir}")
        print(f"Found samples:   {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        # -------------------------
        # Load Sentinel-1 image
        # -------------------------
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # -------------------------
        # Load flood label
        # -------------------------
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        # -------------------------
        # Preprocess
        # -------------------------
        image, valid_mask = preprocess_sentinel1(image)
        label = preprocess_label(label)

        # Ignore pixels where:
        # 1. label is NoData (-1)
        # 2. Sentinel-1 contains invalid values
        label = label.clone()

        ignore_mask = (~valid_mask) | (label == -1)

        label[ignore_mask] = -1

        return image, label


if __name__ == "__main__":

    dataset_root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "sen1floods11"
        / "v1.2"
        / "data"
    )

    dataset = Sen1Floods11Dataset(dataset_root)

    image, label = dataset[0]

    print()
    print("First processed sample:")
    print("-----------------------")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Image dtype:", image.dtype)
    print("Label dtype:", label.dtype)
    print("Image min:", image.min().item())
    print("Image max:", image.max().item())
    print("Image contains NaN:", torch.isnan(image).any().item())
    print("Label values:", torch.unique(label).tolist())
    print("Ignored pixels:", (label == -1).sum().item())
    print("Flood pixels:", (label == 1).sum().item())
    print("Non-flood pixels:", (label == 0).sum().item())