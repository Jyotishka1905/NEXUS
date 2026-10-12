from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "satellite_forecast_subset.csv"
)

CHIPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_satellite_chips"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "models"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_satellite_results.csv"
)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteForecastDataset(Dataset):
    def __init__(self, df, chips_dir):
        self.df = df.reset_index(drop=True)
        self.chips_dir = Path(chips_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        label = int(row["label"])
        lead_time = float(
            row["date_difference_days"]
            if "date_difference_days" in row
            else row["lead_time_days"]
        )

        chip_path = self.chips_dir / f"{sample_id}.pt"
        if not chip_path.exists():
            raise FileNotFoundError(f"Missing satellite chip: {chip_path}")

        image = torch.load(chip_path, weights_only=True)
        return image, label, sample_id, lead_time


class SatelliteForecastClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def evaluate_model(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_lead_times = []

    with torch.no_grad():
        for images, labels, _, lead_times in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_lead_times.extend(lead_times.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.nan_to_num(np.array(all_probs), nan=0.5)
    lead_times = np.array(all_lead_times)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    cm = confusion_matrix(y_true, y_pred)

    # Lead time breakdown
    short_mask = lead_times <= 3.5
    long_mask = lead_times > 3.5

    f1_short = f1_score(y_true[short_mask], y_pred[short_mask], zero_division=0) if np.sum(short_mask) > 0 else 0.0
    f1_long = f1_score(y_true[long_mask], y_pred[long_mask], zero_division=0) if np.sum(long_mask) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "f1_short_lead": f1_short,
        "f1_long_lead": f1_long,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def main():
    print("NEXUS Satellite-Only Future-Event Baseline")
    print("==========================================")

    df = pd.read_csv(SUBSET_FILE)
    print(f"Total samples: {len(df)}")
    print(f"Unique DFO event groups: {df['group_id'].nunique()}")

    # Grouped split: 70% train, 30% temp (15% val, 15% test)
    groups = df["group_id"].to_numpy()
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss_1.split(df, groups=groups))

    temp_groups = groups[temp_idx]
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_sub, test_sub = next(gss_2.split(df.iloc[temp_idx], groups=temp_groups))

    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    print()
    print("Split statistics:")
    print(f"  Train: {len(train_df)} samples, {train_df['group_id'].nunique()} groups")
    print(f"  Val:   {len(val_df)} samples, {val_df['group_id'].nunique()} groups")
    print(f"  Test:  {len(test_df)} samples, {test_df['group_id'].nunique()} groups")

    # Verify zero group leakage
    assert set(train_df["group_id"]).isdisjoint(set(val_df["group_id"]))
    assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"]))
    assert set(val_df["group_id"]).isdisjoint(set(test_df["group_id"]))
    print("Group leakage check: PASSED (zero cross-partition overlap)")

    train_dataset = SatelliteForecastDataset(train_df, CHIPS_DIR)
    val_dataset = SatelliteForecastDataset(val_df, CHIPS_DIR)
    test_dataset = SatelliteForecastDataset(test_df, CHIPS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = SatelliteForecastClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "forecast_satellite_best.pt"

    best_val_f1 = -1.0
    num_epochs = 20

    print()
    print(f"Training on device: {DEVICE} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for images, labels, _, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate_model(model, val_loader, DEVICE)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

    print()
    print(f"Training complete. Best model saved to: {best_model_path}")

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, DEVICE)

    print()
    print("Test Results (Satellite-Only Future-Event Baseline):")
    print("---------------------------------------------------")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1 Score : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"F1 (Lead <= 3 days): {test_metrics['f1_short_lead']:.4f}")
    print(f"F1 (Lead > 3 days) : {test_metrics['f1_long_lead']:.4f}")
    print()
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    results_df = pd.DataFrame(
        [
            {
                "model": "satellite_only_forecasting",
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
                "f1_short_lead": test_metrics["f1_short_lead"],
                "f1_long_lead": test_metrics["f1_long_lead"],
                "test_samples": len(test_df),
            }
        ]
    )
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved evaluation metrics to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
