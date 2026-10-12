from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from torch.utils.data import DataLoader

from multimodal_dataset import MultimodalFloodDataset
from multimodal_model import MultimodalFloodClassifier


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "multimodal_flood_best.pt"
)


def main():

    print("NEXUS Multimodal Evaluation")
    print("===========================")

    print("Device:", DEVICE)
    print("Model:", MODEL_PATH)
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multimodal model not found:\n{MODEL_PATH}"
        )

    test_dataset = MultimodalFloodDataset(
        "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = MultimodalFloodClassifier().to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for (
            satellite,
            environmental,
            targets,
        ) in tqdm(
            test_loader,
            desc="Evaluating",
        ):

            satellite = satellite.to(DEVICE)
            environmental = environmental.to(DEVICE)

            outputs = model(
                satellite,
                environmental,
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = (
                outputs.argmax(dim=1)
                .cpu()
            )

            all_predictions.append(
                predictions
            )

            all_targets.append(
                targets
            )

            all_probabilities.append(
                probabilities.cpu()
            )

    predictions = torch.cat(
        all_predictions
    ).numpy()

    targets = torch.cat(
        all_targets
    ).numpy()

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    if len(np.unique(targets)) == 2:

        auc = roc_auc_score(
            targets,
            probabilities,
        )

    else:

        auc = float("nan")

    print()
    print("Multimodal Test Results")
    print("-----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print()
    print("Test samples:", len(targets))
    print(
        "Positive samples:",
        int(targets.sum()),
    )
    print(
        "Negative samples:",
        int((targets == 0).sum()),
    )

    print()
    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation loss:",
        checkpoint["val_loss"],
    )


if __name__ == "__main__":
    main()