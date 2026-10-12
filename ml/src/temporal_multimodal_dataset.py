from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import Sen1Floods11Dataset


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "temporal_features.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split.csv"
)


# =========================================================
# DATASET
# =========================================================

class TemporalMultimodalFloodDataset(Dataset):

    def __init__(
        self,
        split,
        temporal_scaler=None,
    ):

        self.split = split

        # -------------------------------------------------
        # Load temporal features
        # -------------------------------------------------

        temporal_df = pd.read_csv(
            TEMPORAL_FILE
        )

        temporal_df["chip_id"] = (
            temporal_df["chip_id"]
            .astype(str)
        )

        # -------------------------------------------------
        # Load split information
        # -------------------------------------------------

        split_df = pd.read_csv(
            SPLIT_FILE
        )

        split_df["image"] = (
            split_df["image"]
            .astype(str)
        )

        split_df["chip_id"] = (
            split_df["image"]
            .str.extract(
                r"_(\d+)_S1Hand\.tif$"
            )[0]
            .astype(str)
        )

        # -------------------------------------------------
        # Merge temporal data with split information
        # -------------------------------------------------

        df = temporal_df.merge(
            split_df[
                [
                    "chip_id",
                    "image",
                    "split",
                ]
            ],
            on="chip_id",
            how="inner",
        )

        # -------------------------------------------------
        # Select requested split
        # -------------------------------------------------

        df = df[
            df["split"] == split
        ].copy()

        if len(df) == 0:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )

        # -------------------------------------------------
        # Temporal feature columns
        #
        # 7 days × 4 features = 28 features
        # -------------------------------------------------

        self.feature_columns = [
            column
            for column in df.columns
            if column.startswith("day_")
        ]

        if len(self.feature_columns) != 28:
            raise RuntimeError(
                "Expected 28 temporal features "
                f"but found {len(self.feature_columns)}"
            )

        # -------------------------------------------------
        # Temporal feature matrix
        # -------------------------------------------------

        temporal_values = (
            df[
                self.feature_columns
            ]
            .values
            .astype(np.float32)
        )

        # -------------------------------------------------
        # Check for invalid values
        # -------------------------------------------------

        if not np.isfinite(
            temporal_values
        ).all():

            raise RuntimeError(
                "Temporal features contain "
                "NaN or infinite values."
            )

        # -------------------------------------------------
        # Apply optional scaler
        # -------------------------------------------------

        if temporal_scaler is not None:

            temporal_values = (
                temporal_scaler.transform(
                    temporal_values
                )
                .astype(np.float32)
            )

        # -------------------------------------------------
        # Convert temporal features to tensor
        # -------------------------------------------------

        self.temporal_features = (
            torch.from_numpy(
                temporal_values
            )
        )

        # -------------------------------------------------
        # Targets
        # -------------------------------------------------

        self.targets = torch.tensor(
            df["target"]
            .values
            .astype(np.int64)
        )

        # -------------------------------------------------
        # Satellite dataset
        # -------------------------------------------------

        self.satellite_dataset = (
            Sen1Floods11Dataset(
                DATA_ROOT
            )
        )

        # -------------------------------------------------
        # Map satellite image path → dataset index
        # -------------------------------------------------

        self.path_to_index = {
            str(image_path): index
            for index, (
                image_path,
                _
            ) in enumerate(
                self.satellite_dataset.samples
            )
        }

        # -------------------------------------------------
        # Find satellite indices
        # -------------------------------------------------

        self.satellite_indices = []

        for image_path in df["image"]:

            image_path = str(
                image_path
            )

            if image_path not in self.path_to_index:

                raise RuntimeError(
                    "Satellite image not found:\n"
                    f"{image_path}"
                )

            self.satellite_indices.append(
                self.path_to_index[
                    image_path
                ]
            )

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        self.chip_ids = (
            df["chip_id"]
            .astype(str)
            .tolist()
        )

        self.countries = (
            df["country"]
            .astype(str)
            .tolist()
        )

        # -------------------------------------------------
        # Dataset information
        # -------------------------------------------------

        print(
            f"Temporal multimodal {split} samples:",
            len(df)
        )

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):

        return len(
            self.targets
        )

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, index):

        # -------------------------------------------------
        # Satellite image
        # -------------------------------------------------

        image, _ = (
            self.satellite_dataset[
                self.satellite_indices[index]
            ]
        )

        # -------------------------------------------------
        # Temporal environmental data
        # -------------------------------------------------

        temporal = (
            self.temporal_features[
                index
            ]
        )

        # -------------------------------------------------
        # Classification target
        # -------------------------------------------------

        target = (
            self.targets[
                index
            ]
        )

        return (
            image,
            temporal,
            target,
        )


# =========================================================
# DATALOADERS
# =========================================================

def create_temporal_multimodal_dataloaders(
    batch_size=2,
    temporal_scaler=None,
):

    # -----------------------------------------------------
    # Create datasets
    # -----------------------------------------------------

    train_dataset = (
        TemporalMultimodalFloodDataset(
            split="train",
            temporal_scaler=temporal_scaler,
        )
    )

    val_dataset = (
        TemporalMultimodalFloodDataset(
            split="val",
            temporal_scaler=temporal_scaler,
        )
    )

    test_dataset = (
        TemporalMultimodalFloodDataset(
            split="test",
            temporal_scaler=temporal_scaler,
        )
    )

    # -----------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "NEXUS Temporal Multimodal Dataset Test"
    )

    print(
        "======================================="
    )

    print()

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_temporal_multimodal_dataloaders(
        batch_size=2
    )

    # -----------------------------------------------------
    # Get first training batch
    # -----------------------------------------------------

    image, temporal, target = next(
        iter(train_loader)
    )

    print()

    print(
        "First training batch:"
    )

    print(
        "Satellite shape:",
        image.shape
    )

    print(
        "Temporal shape:",
        temporal.shape
    )

    print(
        "Target shape:",
        target.shape
    )

    print(
        "Satellite dtype:",
        image.dtype
    )

    print(
        "Temporal dtype:",
        temporal.dtype
    )

    print(
        "Target dtype:",
        target.dtype
    )

    print(
        "Temporal NaN:",
        torch.isnan(
            temporal
        ).any().item()
    )

    print(
        "Temporal Inf:",
        torch.isinf(
            temporal
        ).any().item()
    )

    print(
        "Target values:",
        target.unique().tolist()
    )

    print()

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print(
        "Test batches:",
        len(test_loader)
    )