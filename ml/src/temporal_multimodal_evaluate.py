from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from temporal_multimodal_dataset import (
    create_temporal_multimodal_dataloaders,
)
from temporal_multimodal_model import (
    TemporalMultimodalFloodClassifier,
)


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
    / "temporal_multimodal_improved_best.pt"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "NEXUS Improved Temporal Multimodal Evaluation"
    )

    print(
        "================================================"
    )

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    _, _, test_loader = (
        create_temporal_multimodal_dataloaders(
            batch_size=2
        )
    )

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    model = (
        TemporalMultimodalFloodClassifier()
        .to(DEVICE)
    )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for (
            satellite,
            temporal,
            labels,
        ) in tqdm(
            test_loader,
            desc="Evaluating",
        ):

            satellite = satellite.to(
                DEVICE
            )

            temporal = temporal.to(
                DEVICE
            )

            outputs = model(
                satellite,
                temporal,
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = (
                outputs.argmax(dim=1)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_targets.extend(
                labels.numpy()
            )

    predictions = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

    targets = np.array(
        all_targets
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print()

    print(
        "Improved Temporal Multimodal "
        "Test Results"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print()

    print(
        "Test samples:",
        len(targets),
    )

    print(
        "Positive samples:",
        int(targets.sum()),
    )

    print(
        "Negative samples:",
        int(
            (targets == 0).sum()
        ),
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

    print()

    print(
        "Evaluation complete."
    )


if __name__ == "__main__":

    main()