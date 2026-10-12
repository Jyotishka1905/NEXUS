from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COVERAGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "sentinel1_coverage.csv"
)


def main():

    df = pd.read_csv(COVERAGE_FILE)

    print("NEXUS Sentinel-1 Coverage Analysis")
    print("==================================")

    # Extract event group.
    df["group_id"] = (
        df["sample_id"]
        .str.replace(
            "event_",
            "event_",
            regex=False,
        )
        .str.replace(
            "control_",
            "event_",
            regex=False,
        )
    )

    # Only actual coverage records.
    covered = df[df["coverage"] == 1].copy()

    # Count covered samples within each event group.
    pair_status = (
        df.groupby("group_id")
        .agg(
            samples=("sample_id", "count"),
            covered=("coverage", lambda x: int((x == 1).sum())),
        )
        .reset_index()
    )

    pair_status["complete_pair"] = (
        pair_status["covered"] == 2
    )

    print()
    print("Event-pair coverage")
    print("-------------------")

    print(
        pair_status["complete_pair"]
        .value_counts()
    )

    complete_groups = set(
        pair_status.loc[
            pair_status["complete_pair"],
            "group_id",
        ]
    )

    complete = covered[
        covered["group_id"].isin(
            complete_groups
        )
    ].copy()

    print()
    print(
        f"Complete event/control pairs: "
        f"{len(complete_groups)}"
    )

    print(
        f"Samples in complete pairs: "
        f"{len(complete)}"
    )

    print()
    print("Complete pairs by country:")

    print(
        complete.groupby("country")["group_id"]
        .nunique()
    )

    print()
    print("Complete-pair labels:")

    print(
        complete["label"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Acquisition timing
    # ---------------------------------------------------------

    if "date_difference_days" in complete.columns:

        print()
        print("Satellite acquisition timing")
        print("-----------------------------")

        print(
            complete[
                "date_difference_days"
            ].describe()
        )

    # ---------------------------------------------------------
    # Print incomplete groups
    # ---------------------------------------------------------

    incomplete = pair_status[
        ~pair_status["complete_pair"]
    ]

    print()
    print(
        f"Incomplete pairs: "
        f"{len(incomplete)}"
    )

    if len(incomplete) > 0:

        print()
        print(
            incomplete.to_string(
                index=False
            )
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()