from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COVERAGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "sentinel1_forecast_coverage.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "environmental"
    / "processed"
    / "satellite_forecast_subset.csv"
)


def main():

    print("NEXUS Forecast-Valid Satellite Subset")
    print("======================================")

    df = pd.read_csv(
        COVERAGE_FILE,
        parse_dates=[
            "forecast_date",
            "acquisition_date",
        ],
    )

    # event_4361 and control_4361 belong to the same
    # independent DFO event group.
    df["group_id"] = (
        df["sample_id"]
        .str.replace(
            "control_",
            "event_",
            regex=False,
        )
    )

    covered = df[df["coverage"] == 1].copy()

    # A usable forecasting pair requires BOTH:
    #   event sample
    #   control sample
    coverage_counts = (
        covered.groupby("group_id")
        .size()
    )

    complete_groups = set(
        coverage_counts[
            coverage_counts == 2
        ].index
    )

    subset = covered[
        covered["group_id"].isin(
            complete_groups
        )
    ].copy()

    subset = subset.sort_values(
        ["group_id", "label"]
    ).reset_index(drop=True)

    subset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Coverage analysis")
    print("------------------")

    print(
        f"Total event groups: "
        f"{df['group_id'].nunique()}"
    )

    print(
        f"Complete event/control pairs: "
        f"{len(complete_groups)}"
    )

    print(
        f"Usable samples: "
        f"{len(subset)}"
    )

    print()
    print("Labels:")
    print(
        subset["label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Complete pairs by country:")
    print(
        subset.groupby("country")["group_id"]
        .nunique()
    )

    print()
    print("Acquisition lead time:")
    print(
        subset[
            "date_difference_days"
        ].describe()
    )

    print()
    print("Forecast-valid satellite products:")
    print(
        subset[
            [
                "sample_id",
                "label",
                "group_id",
                "product_name",
                "product_id",
                "forecast_date",
                "acquisition_date",
                "date_difference_days",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()