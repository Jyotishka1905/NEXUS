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


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "temporal_features.csv"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("NEXUS Temporal Environmental Baseline")
print("=====================================")
print()

print("Feature file:")
print(FEATURE_FILE)
print()

df = pd.read_csv(FEATURE_FILE)

split_df = pd.read_csv(SPLIT_FILE)


# =========================================================
# PREPARE CHIP ID FOR SPLIT MATCHING
# =========================================================

split_df["image"] = (
    split_df["image"]
    .astype(str)
)

split_df["chip_id"] = (
    split_df["image"]
    .str.extract(
        r"_(\d+)_S1Hand\.tif$"
    )[0]
    .astype(str)
)

df["chip_id"] = (
    df["chip_id"]
    .astype(str)
)


# =========================================================
# MERGE SPLIT INFORMATION
# =========================================================

df = df.merge(
    split_df[
        [
            "chip_id",
            "split",
        ]
    ],
    on="chip_id",
    how="inner",
)


# =========================================================
# CHECK DATA
# =========================================================

print("Samples:", len(df))
print()

print("Split distribution:")
print(
    df["split"].value_counts()
)
print()


print("Target distribution:")
print(
    df["target"].value_counts()
)
print()


# =========================================================
# TEMPORAL FEATURES
# =========================================================

feature_columns = [
    column
    for column in df.columns
    if column.startswith("day_")
]


print(
    "Temporal feature count:",
    len(feature_columns)
)

print()


if len(feature_columns) != 28:

    raise RuntimeError(
        "Expected 28 temporal features "
        "(7 days × 4 features), but found "
        f"{len(feature_columns)}."
    )


# =========================================================
# CHECK MISSING VALUES
# =========================================================

missing_values = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


if missing_values > 0:

    raise RuntimeError(
        f"Temporal features contain "
        f"{missing_values} missing values."
    )


# =========================================================
# TRAIN / VALIDATION / TEST
# =========================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print()


# =========================================================
# FEATURES AND TARGET
# =========================================================

X_train = (
    train_df[
        feature_columns
    ]
    .values
)

y_train = (
    train_df["target"]
    .values
)

X_val = (
    val_df[
        feature_columns
    ]
    .values
)

y_val = (
    val_df["target"]
    .values
)

X_test = (
    test_df[
        feature_columns
    ]
    .values
)

y_test = (
    test_df["target"]
    .values
)


# =========================================================
# STANDARDIZATION
#
# Fit ONLY on training data.
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)


# =========================================================
# RANDOM FOREST
# =========================================================

print("Training temporal environmental model...")
print()

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_features="sqrt",
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# =========================================================
# VALIDATION
# =========================================================

val_predictions = model.predict(
    X_val
)

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


val_accuracy = accuracy_score(
    y_val,
    val_predictions,
)

val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0,
)

val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0,
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0,
)


if len(np.unique(y_val)) == 2:

    val_auc = roc_auc_score(
        y_val,
        val_probabilities,
    )

else:

    val_auc = float("nan")


# =========================================================
# TEST
# =========================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


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


# =========================================================
# RESULTS
# =========================================================

print()
print("Temporal Environmental Validation Results")
print("------------------------------------------")
print(
    f"Accuracy : {val_accuracy:.4f}"
)
print(
    f"Precision: {val_precision:.4f}"
)
print(
    f"Recall   : {val_recall:.4f}"
)
print(
    f"F1 Score : {val_f1:.4f}"
)
print(
    f"ROC-AUC  : {val_auc:.4f}"
)


print()
print("Temporal Environmental Test Results")
print("------------------------------------")
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
print("Test samples:", len(y_test))
print(
    "Positive samples:",
    int(y_test.sum()),
)
print(
    "Negative samples:",
    int(
        (y_test == 0).sum()
    ),
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print()
print("Top temporal features")
print("---------------------")

for _, row in importance.head(10).iterrows():

    print(
        f"{row['feature']}: "
        f"{row['importance']:.4f}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results_file = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "temporal_baseline_results.csv"
)

results = pd.DataFrame(
    [
        {
            "model": "temporal_environmental",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "test_samples": len(y_test),
            "positive_samples": int(
                y_test.sum()
            ),
            "negative_samples": int(
                (y_test == 0).sum()
            ),
        }
    ]
)

results.to_csv(
    results_file,
    index=False,
)


print()
print(
    "Results saved to:"
)
print(results_file)

print()
print(
    "Temporal environmental baseline complete."
)