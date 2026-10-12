from pathlib import Path
import csv

from torch.utils.data import DataLoader, Subset

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


def load_split_indices():
    split_indices = {
        "train": [],
        "val": [],
        "test": [],
    }

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    # Dataset ordering is deterministic because dataset.py
    # sorts the image files.
    dataset = Sen1Floods11Dataset(DATA_ROOT)

    path_to_index = {
        str(image_path): index
        for index, (image_path, _) in enumerate(dataset.samples)
    }

    for row in rows:

        image_path = row["image"]

        if image_path not in path_to_index:
            raise RuntimeError(
                f"Image from split.csv not found in dataset: {image_path}"
            )

        index = path_to_index[image_path]

        split = row["split"]

        split_indices[split].append(index)

    return dataset, split_indices


def create_dataloaders(batch_size=4):

    dataset, split_indices = load_split_indices()

    train_dataset = Subset(
        dataset,
        split_indices["train"]
    )

    val_dataset = Subset(
        dataset,
        split_indices["val"]
    )

    test_dataset = Subset(
        dataset,
        split_indices["test"]
    )

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

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print()
    print("DataLoaders created successfully.")
    print("---------------------------------")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print()
    print("First training batch:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Label values:", labels.unique().tolist())
