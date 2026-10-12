from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "forecast_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
)

RESULTS_FILE = OUTPUT_DIR / "forecast_environmental_results.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "forecast_feature_importance.csv"


RANDOM_STATE = 42


def make_group_id(sample_id):
    """
    event_4361 -> event_4361
    control_4361 -> event_4361

    This ensures the positive/control samples associated
    with the same DFO event stay in the same split.
    """
    return "event_" + str(sample_id).split("_")[-1]


def print_metrics(name, y_true, probabilities, threshold=0.5):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    print()
    print(name)
    print("-" * len(name))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print()
    print("Confusion matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():

    print("NEXUS Environmental Future-Event Baseline")
    print("==========================================")

    df = pd.read_csv(FEATURE_FILE)

    print(f"Samples: {len(df)}")

    # ---------------------------------------------------------
    # Construct event groups
    # ---------------------------------------------------------

    df["group_id"] = df["sample_id"].apply(
        make_group_id
    )

    print(
        f"Unique DFO event groups: "
        f"{df['group_id'].nunique()}"
    )

    print()
    print("Labels:")
    print(df["label"].value_counts().sort_index())

    # ---------------------------------------------------------
    # Select numerical model features
    # ---------------------------------------------------------

    excluded_columns = {
        "sample_id",
        "label",
        "country",
        "latitude",
        "longitude",
        "forecast_date",
        "group_id",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    X = df[feature_columns].copy()
    y = df["label"].to_numpy()
    groups = df["group_id"].to_numpy()

    print()
    print(f"Model features: {len(feature_columns)}")

    # ---------------------------------------------------------
    # Group-aware train / validation / test split
    # ---------------------------------------------------------

    # First split: 70% train, 30% temporary.
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            X,
            y,
            groups=groups,
        )
    )

    # Second split: temporary -> validation/test.
    temp_groups = groups[temp_idx]

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    val_relative_idx, test_relative_idx = next(
        splitter_2.split(
            X.iloc[temp_idx],
            y[temp_idx],
            groups=temp_groups,
        )
    )

    val_idx = temp_idx[val_relative_idx]
    test_idx = temp_idx[test_relative_idx]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    groups_train = groups[train_idx]
    groups_val = groups[val_idx]
    groups_test = groups[test_idx]

    print()
    print("Split")
    print("-----")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")
    print(f"Test samples:  {len(test_idx)}")

    print()
    print(f"Train event groups: {len(np.unique(groups_train))}")
    print(f"Val event groups:   {len(np.unique(groups_val))}")
    print(f"Test event groups:  {len(np.unique(groups_test))}")

    # Verify no event appears in multiple splits.
    train_groups = set(groups_train)
    val_groups = set(groups_val)
    test_groups = set(groups_test)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print()
    print("Group leakage check: PASSED")

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print()
    print("Training Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    val_metrics = print_metrics(
        "Validation results",
        y_val,
        val_probabilities,
    )

    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = print_metrics(
        "Test results",
        y_test,
        test_probabilities,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("Top 15 features")
    print("---------------")

    print(
        importance.head(15).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        [
            {
                "model": "environmental_random_forest",
                "split": "validation",
                **val_metrics,
                "samples": len(y_val),
                "positive_samples": int(y_val.sum()),
                "negative_samples": int((y_val == 0).sum()),
            },
            {
                "model": "environmental_random_forest",
                "split": "test",
                **test_metrics,
                "samples": len(y_test),
                "positive_samples": int(y_test.sum()),
                "negative_samples": int((y_test == 0).sum()),
            },
        ]
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print()
    print("Saved:")
    print(RESULTS_FILE)
    print(IMPORTANCE_FILE)

    print()
    print("Done.")


if __name__ == "__main__":
    main()