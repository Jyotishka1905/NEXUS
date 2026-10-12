from pathlib import Path

import torch
from tqdm import tqdm

from dataloaders import create_dataloaders
from model import SatelliteUNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "satellite_unet_best.pt"
)


def calculate_metrics(predictions, targets):
    valid = targets != -1

    predictions = predictions[valid]
    targets = targets[valid]

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    return accuracy, precision, recall, f1, iou


def main():
    print("NEXUS Satellite-Only Evaluation")
    print("===============================")
    print("Device:", DEVICE)
    print("Model:", MODEL_PATH)
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    _, _, test_loader = create_dataloaders(batch_size=2)

    model = SatelliteUNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.append(predictions)
            all_targets.append(labels)

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    accuracy, precision, recall, f1, iou = calculate_metrics(
        predictions,
        targets,
    )

    print()
    print("Satellite-Only Test Results")
    print("---------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")

    print()
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["val_loss"])


if __name__ == "__main__":
    main()
    