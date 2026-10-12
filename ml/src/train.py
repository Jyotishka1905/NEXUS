from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloaders import create_dataloaders
from model import SatelliteUNet


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "satellite_unet_best.pt"


# -------------------------
# Training
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader)


# -------------------------
# Validation
# -------------------------

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Main
# -------------------------

def main():

    print("NEXUS Satellite-Only Training")
    print("==============================")
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print()

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = SatelliteUNet().to(DEVICE)

    # -1 = NoData pixels.
    # These pixels are excluded from the loss.
    criterion = nn.CrossEntropyLoss(
        ignore_index=-1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        print(
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
                BEST_MODEL_PATH
            )

            print("✓ Best model saved.")


    print()
    print("Training complete.")
    print("Best validation loss:", best_val_loss)
    print("Model saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()