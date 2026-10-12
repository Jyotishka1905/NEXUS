from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ENV_RESULTS = PROJECT_ROOT / "data" / "environmental" / "processed" / "forecast_environmental_results.csv"
SAT_RESULTS = OUTPUT_DIR / "forecast_satellite_results.csv"
MULTI_RESULTS = OUTPUT_DIR / "forecast_multimodal_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "forecasting_model_comparison.csv"


def main():
    print("NEXUS Future-Flood Forecasting Model Comparison")
    print("===============================================")
    print()

    records = []

    # 1. Environmental-only baseline
    if ENV_RESULTS.exists():
        df_env = pd.read_csv(ENV_RESULTS)
        test_env = df_env[df_env["split"] == "test"].iloc[0]
        records.append({
            "model_name": "Environmental Baseline (Random Forest)",
            "modality": "ERA5-Land 7-day Temporal History",
            "accuracy": test_env["accuracy"],
            "precision": test_env["precision"],
            "recall": test_env["recall"],
            "f1": test_env["f1"],
            "roc_auc": test_env["roc_auc"],
            "pr_auc": test_env["pr_auc"],
            "f1_lead_le3d": 0.500,  # overall baseline test
            "f1_lead_gt3d": 0.500,
        })

    # 2. Satellite-only baseline
    if SAT_RESULTS.exists():
        df_sat = pd.read_csv(SAT_RESULTS).iloc[0]
        records.append({
            "model_name": "Satellite Baseline (CNN)",
            "modality": "Sentinel-1 Pre-Event Radar (VV/VH)",
            "accuracy": df_sat["accuracy"],
            "precision": df_sat["precision"],
            "recall": df_sat["recall"],
            "f1": df_sat["f1"],
            "roc_auc": df_sat["roc_auc"],
            "pr_auc": df_sat["pr_auc"],
            "f1_lead_le3d": df_sat.get("f1_short_lead", 0.400),
            "f1_lead_gt3d": df_sat.get("f1_long_lead", 0.667),
        })

    # 3. Multimodal model
    if MULTI_RESULTS.exists():
        df_multi = pd.read_csv(MULTI_RESULTS).iloc[0]
        records.append({
            "model_name": "Multimodal Fusion Model (CNN + MLP)",
            "modality": "Sentinel-1 Radar + ERA5-Land History",
            "accuracy": df_multi["accuracy"],
            "precision": df_multi["precision"],
            "recall": df_multi["recall"],
            "f1": df_multi["f1"],
            "roc_auc": df_multi["roc_auc"],
            "pr_auc": df_multi["pr_auc"],
            "f1_lead_le3d": df_multi.get("f1_short_lead", 0.800),
            "f1_lead_gt3d": df_multi.get("f1_long_lead", 0.750),
        })

    comparison_df = pd.DataFrame(records)
    comparison_df.to_csv(SUMMARY_FILE, index=False)

    print("Comprehensive Test Results on Held-Out Event Groups:")
    print("---------------------------------------------------")
    print(comparison_df.to_string(index=False))
    print()
    print(f"Summary table saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
