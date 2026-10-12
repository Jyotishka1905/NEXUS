from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from dataset import Sen1Floods11Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "environmental_features.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11"
    / "v1.2"
    / "data"
)


# --------------------------------------------------
# Load environmental features
# --------------------------------------------------

df = pd.read_csv(ENV_FILE)

print("NEXUS Environmental-Only Baseline")
print("=================================")
print("Samples:", len(df))
print()


# --------------------------------------------------
# Load split information
# --------------------------------------------------

split_df = pd.read_csv(SPLIT_FILE)

split_df["image"] = split_df["image"].astype(str)

split_df["chip_id"] = (
    split_df["image"]
    .str.extract(r"_(\d+)_S1Hand\.tif$")[0]
)

split_df["chip_id"] = split_df["chip_id"].astype(str)
df["chip_id"] = df["chip_id"].astype(str)


# Keep image path as well as split.
df = df.merge(
    split_df[["chip_id", "image", "split"]],
    on="chip_id",
    how="inner",
)


print("Split distribution:")
print(df["split"].value_counts())
print()


# --------------------------------------------------
# Load Sen1Floods11 dataset
# --------------------------------------------------

dataset = Sen1Floods11Dataset(DATA_ROOT)

path_to_index = {
    str(image_path): index
    for index, (image_path, _) in enumerate(dataset.samples)
}


# --------------------------------------------------
# Create chip-level flood target
# --------------------------------------------------

targets = []

for _, row in df.iterrows():

    image_path = row["image"]

    if image_path not in path_to_index:
        raise RuntimeError(
            f"Could not find dataset image:\n{image_path}"
        )

    index = path_to_index[image_path]

    _, label = dataset[index]

    valid = label != -1

    valid_pixels = valid.sum().item()

    if valid_pixels == 0:
        flood_fraction = 0.0
    else:
        flood_fraction = (
            (label[valid] == 1).sum().item()
            / valid_pixels
        )

    # Chip is considered flood-positive when
    # at least 1% of valid pixels are flood.
    target = 1 if flood_fraction >= 0.01 else 0

    targets.append(target)


df["target"] = targets


print("Chip-level target distribution:")
print(df["target"].value_counts())
print()


# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "temperature_c",
    "precipitation_mm",
]


# --------------------------------------------------
# Train / validation / test
# --------------------------------------------------

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]


X_train = train_df[features].values
y_train = train_df["target"].values

X_val = val_df[features].values
y_val = val_df["target"].values

X_test = test_df[features].values
y_test = test_df["target"].values


print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print()


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

if len(np.unique(y_test)) == 2:

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

else:

    auc = float("nan")


print("Environmental-Only Test Results")
print("--------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print()

print("Test samples:", len(y_test))
print("Positive samples:", int(y_test.sum()))
print("Negative samples:", int((y_test == 0).sum()))