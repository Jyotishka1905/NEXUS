from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from dataset import Sen1Floods11Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class SatelliteChipDataset(Dataset):

    def __init__(self, base_dataset, indices):

        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        dataset_index = self.indices[index]

        image, label = self.base_dataset[dataset_index]

        valid = label != -1

        if valid.sum() == 0:
            target = 0
        else:
            flood_fraction = (
                (label[valid] == 1).sum().item()
                / valid.sum().item()
            )

            target = 1 if flood_fraction >= 0.01 else 0

        return image, torch.tensor(target, dtype=torch.long)


class SatelliteClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


def load_splits(dataset):

    split_df = pd.read_csv(SPLIT_FILE)

    path_to_index = {
        str(image_path): index
        for index, (image_path, _) in enumerate(
            dataset.samples
        )
    }

    split_indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    for _, row in split_df.iterrows():

        image_path = str(row["image"])

        if image_path not in path_to_index:
            raise RuntimeError(
                f"Image not found:\n{image_path}"
            )

        split_indices[row["split"]].append(
            path_to_index[image_path]
        )

    return split_indices


def main():

    print("NEXUS Satellite-Only Chip Classifier")
    print("=====================================")
    print("Device:", DEVICE)
    print()

    base_dataset = Sen1Floods11Dataset(DATA_ROOT)

    split_indices = load_splits(base_dataset)

    train_dataset = SatelliteChipDataset(
        base_dataset,
        split_indices["train"],
    )

    val_dataset = SatelliteChipDataset(
        base_dataset,
        split_indices["val"],
    )

    test_dataset = SatelliteChipDataset(
        base_dataset,
        split_indices["test"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))
    print()

    model = SatelliteClassifier().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    best_val_loss = float("inf")

    model_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir
        / "satellite_chip_best.pt"
    )

    for epoch in range(1, 11):

        model.train()

        train_loss = 0.0

        for images, targets in train_loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, targets in val_loader:

                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets,
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch}/10 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                model_path,
            )

    print()
    print("Training complete.")
    print("Best validation loss:", best_val_loss)
    print("Model saved to:", model_path)

    # -------------------------
    # Test evaluation
    # -------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, targets in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = outputs.argmax(
                dim=1
            )

            all_targets.extend(
                targets.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    auc = roc_auc_score(
        all_targets,
        all_probabilities,
    )

    print()
    print("Satellite-Only Chip Test Results")
    print("---------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print()
    print("Test samples:", len(all_targets))
    print(
        "Positive samples:",
        sum(all_targets),
    )
    print(
        "Negative samples:",
        len(all_targets) - sum(all_targets),
    )


if __name__ == "__main__":
    main()