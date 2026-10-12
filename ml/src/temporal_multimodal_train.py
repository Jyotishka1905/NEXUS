from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from temporal_multimodal_dataset import (
    create_temporal_multimodal_dataloaders,
)
from temporal_multimodal_model import (
    TemporalMultimodalFloodClassifier,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "temporal_multimodal_improved_best.pt"
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):
    model.train()

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for satellite, temporal, labels in progress:

        satellite = satellite.to(DEVICE)
        temporal = temporal.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            satellite,
            temporal,
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(loader)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for satellite, temporal, labels in loader:

            satellite = satellite.to(DEVICE)
            temporal = temporal.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                satellite,
                temporal,
            )

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("NEXUS Improved Temporal Multimodal Training")
    print("============================================")
    print()
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Early stopping patience:", PATIENCE)
    print()

    (
        train_loader,
        val_loader,
        _,
    ) = create_temporal_multimodal_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = TemporalMultimodalFloodClassifier().to(
        DEVICE
    )

    # -----------------------------------------------------
    # Calculate class weights from training labels
    # -----------------------------------------------------

    train_labels = []

    for _, _, labels in train_loader:

        train_labels.extend(
            labels.cpu().numpy().tolist()
        )

    train_labels = np.array(
        train_labels,
        dtype=np.int64,
    )

    class_counts = np.bincount(
        train_labels,
        minlength=2,
    )

    total = class_counts.sum()

    class_weights = total / (
        2.0 * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Class counts:", class_counts.tolist())
    print(
        "Class weights:",
        class_weights.cpu().tolist(),
    )
    print()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "class_weights": class_weights.cpu(),
                },
                BEST_MODEL_PATH,
            )

            print("✓ Best model saved.")

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                PATIENCE,
            )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    print()
    print("Training complete.")
    print(
        "Best validation loss:",
        best_val_loss,
    )
    print(
        "Model saved to:",
        BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()