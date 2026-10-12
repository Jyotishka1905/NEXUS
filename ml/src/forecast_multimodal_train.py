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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUBSET_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "satellite_forecast_subset.csv"
)

ENV_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_features.csv"
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
    / "forecast_multimodal_results.csv"
)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalForecastDataset(Dataset):
    def __init__(self, df, env_features_matrix, chips_dir):
        self.df = df.reset_index(drop=True)
        self.env_features = torch.from_numpy(env_features_matrix).float()
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

        satellite_tensor = torch.load(chip_path, weights_only=True)
        env_tensor = self.env_features[idx]

        return satellite_tensor, env_tensor, label, sample_id, lead_time


class MultimodalFutureFloodModel(nn.Module):
    def __init__(self, env_dim, num_classes=2):
        super().__init__()

        # 1. Satellite Radar Encoder
        self.satellite_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
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
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # 2. Environmental Temporal History Encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 3. Multimodal Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, satellite, env):
        sat_emb = self.satellite_encoder(satellite)
        env_emb = self.env_encoder(env)
        fused = torch.cat([sat_emb, env_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits


def evaluate_model(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_lead_times = []

    with torch.no_grad():
        for sat, env, labels, _, lead_times in loader:
            sat = sat.to(device)
            env = env.to(device)
            logits = model(sat, env)

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
    print("NEXUS Multimodal Future-Flood Forecasting Model")
    print("===============================================")

    # 1. Load satellite subset & environmental features
    sat_df = pd.read_csv(SUBSET_FILE)
    env_df = pd.read_csv(ENV_FEATURES_FILE)

    excluded = {"sample_id", "label", "country", "latitude", "longitude", "forecast_date", "group_id"}
    env_cols = [c for c in env_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(env_df[c])]

    # Merge to align sample order perfectly
    merged = sat_df[["sample_id", "label", "country", "latitude", "longitude", "forecast_date", "group_id", "date_difference_days"]].merge(
        env_df[["sample_id"] + env_cols], on="sample_id"
    )

    print(f"Aligned samples: {len(merged)}")
    print(f"Environmental features: {len(env_cols)}")
    print(f"Event groups: {merged['group_id'].nunique()}")

    # 2. GroupShuffleSplit matching exactly the satellite-only & environmental-only baselines
    groups = merged["group_id"].to_numpy()
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss_1.split(merged, groups=groups))

    temp_groups = groups[temp_idx]
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_sub, test_sub = next(gss_2.split(merged.iloc[temp_idx], groups=temp_groups))

    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    train_df = merged.iloc[train_idx].copy()
    val_df = merged.iloc[val_idx].copy()
    test_df = merged.iloc[test_idx].copy()

    # Normalize environmental features using training statistics
    scaler = StandardScaler()
    X_env_train = scaler.fit_transform(train_df[env_cols].fillna(0).to_numpy())
    X_env_val = scaler.transform(val_df[env_cols].fillna(0).to_numpy())
    X_env_test = scaler.transform(test_df[env_cols].fillna(0).to_numpy())

    train_dataset = MultimodalForecastDataset(train_df, X_env_train, CHIPS_DIR)
    val_dataset = MultimodalForecastDataset(val_df, X_env_val, CHIPS_DIR)
    test_dataset = MultimodalForecastDataset(test_df, X_env_test, CHIPS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print()
    print("Partitions:")
    print(f"  Train: {len(train_df)} samples, {train_df['group_id'].nunique()} groups")
    print(f"  Val:   {len(val_df)} samples, {val_df['group_id'].nunique()} groups")
    print(f"  Test:  {len(test_df)} samples, {test_df['group_id'].nunique()} groups")

    # 3. Model setup
    model = MultimodalFutureFloodModel(env_dim=len(env_cols)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "forecast_multimodal_best.pt"

    best_val_f1 = -1.0
    num_epochs = 25

    print()
    print(f"Training Multimodal Model on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for sat, env, labels, _, _ in train_loader:
            sat = sat.to(DEVICE)
            env = env.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sat, env)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate_model(model, val_loader, DEVICE)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "env_scaler_mean": scaler.mean_,
                    "env_scaler_scale": scaler.scale_,
                    "env_feature_names": env_cols,
                },
                best_model_path,
            )

        if epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

    print()
    print(f"Training complete. Best model checkpoint saved to: {best_model_path}")

    # 4. Final test evaluation with best checkpoint
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, DEVICE)

    print()
    print("Test Results (Multimodal Future-Event Forecasting Model):")
    print("---------------------------------------------------------")
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
                "model": "multimodal_forecasting",
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
