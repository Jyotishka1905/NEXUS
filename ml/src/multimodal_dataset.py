from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import Sen1Floods11Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "environmental_features.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)


class MultimodalFloodDataset(Dataset):

    def __init__(self, split):
        self.split = split

        # Load environmental features
        env_df = pd.read_csv(ENV_FILE)

        # Load train/validation/test split
        split_df = pd.read_csv(SPLIT_FILE)

        # Extract chip ID from image filename
        split_df["chip_id"] = (
            split_df["image"]
            .astype(str)
            .str.extract(r"_(\d+)_S1Hand\.tif$")[0]
            .astype(str)
        )

        env_df["chip_id"] = env_df["chip_id"].astype(str)

        # Combine environmental data with split information
        df = env_df.merge(
            split_df[["chip_id", "image", "split"]],
            on="chip_id",
            how="inner",
        )

        # Keep only requested split
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )

        self.df = df

        # Existing Sentinel-1 dataset
        self.satellite_dataset = Sen1Floods11Dataset(DATA_ROOT)

        self.path_to_index = {
            str(image_path): index
            for index, (image_path, _) in enumerate(
                self.satellite_dataset.samples
            )
        }

        print(f"Multimodal {split} samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image"]

        if image_path not in self.path_to_index:
            raise RuntimeError(
                f"Satellite image not found:\n{image_path}"
            )

        satellite_index = self.path_to_index[image_path]

        satellite_image, label = self.satellite_dataset[
            satellite_index
        ]

        # Environmental features
        environmental = torch.tensor(
            [
                float(row["temperature_c"]),
                float(row["precipitation_mm"]),
            ],
            dtype=torch.float32,
        )

        # Same chip-level target used by the environmental baseline
        valid = label != -1

        valid_pixels = valid.sum().item()

        if valid_pixels == 0:
            flood_fraction = 0.0
        else:
            flood_fraction = (
                (label[valid] == 1).sum().item()
                / valid_pixels
            )

        target = 1 if flood_fraction >= 0.01 else 0

        target = torch.tensor(
            target,
            dtype=torch.long,
        )

        return satellite_image, environmental, target


if __name__ == "__main__":

    print("NEXUS Multimodal Dataset")
    print("========================")

    train_dataset = MultimodalFloodDataset("train")
    val_dataset = MultimodalFloodDataset("val")
    test_dataset = MultimodalFloodDataset("test")

    print()
    print("Dataset sizes:")
    print("Train:", len(train_dataset))
    print("Val:  ", len(val_dataset))
    print("Test: ", len(test_dataset))

    satellite, environmental, target = train_dataset[0]

    print()
    print("First sample:")
    print("------------")
    print("Satellite shape:", satellite.shape)
    print("Satellite dtype:", satellite.dtype)
    print("Environmental:", environmental)
    print("Environmental shape:", environmental.shape)
    print("Target:", target.item())
    print("Target dtype:", target.dtype)