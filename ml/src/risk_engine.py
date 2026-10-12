from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import pandas as pd

from forecast_multimodal_train import MultimodalFutureFloodModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "models"
CHECKPOINT_PATH = MODELS_DIR / "forecast_multimodal_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NexusRiskEngine:
    """
    Regional Risk & Early Warning Engine for NEXUS.
    Translates multimodal model forecasts into regional risk scores,
    threat levels, and automated early warning alerts.
    """

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.feature_names = []
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
        self.scaler_mean = checkpoint["env_scaler_mean"]
        self.scaler_scale = checkpoint["env_scaler_scale"]
        self.feature_names = checkpoint["env_feature_names"]

        self.model = MultimodalFutureFloodModel(env_dim=len(self.feature_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    def assess_risk(
        self,
        region_name: str,
        country: str,
        latitude: float,
        longitude: float,
        forecast_date: str,
        lead_time_days: float,
        satellite_tensor: torch.Tensor,
        env_feature_values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compute flood probability, risk category, factor attribution, and alerts.
        """
        # Vectorize environmental features matching exact training scaler order
        feat_vector = np.array(
            [env_feature_values.get(col, 0.0) for col in self.feature_names],
            dtype=np.float32,
        )

        norm_feat = (feat_vector - self.scaler_mean) / (self.scaler_scale + 1e-7)
        norm_feat_tensor = torch.from_numpy(norm_feat).float().unsqueeze(0).to(DEVICE)

        if satellite_tensor.dim() == 3:
            satellite_tensor = satellite_tensor.unsqueeze(0)
        satellite_tensor = satellite_tensor.to(DEVICE)

        with torch.no_grad():
            logits = self.model(satellite_tensor, norm_feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            flood_prob = float(probs[1])

        # Map calibrated probability to operational risk levels
        if flood_prob >= 0.80:
            risk_level = "SEVERE"
            risk_color = "#ef4444"  # Red
            alert_action = "CRITICAL WARNING: Imminent widespread inundation. Trigger emergency response and evacuate vulnerable zones."
        elif flood_prob >= 0.60:
            risk_level = "HIGH"
            risk_color = "#f97316"  # Orange
            alert_action = "FLOOD WARNING: Elevated river discharge and severe surface runoff expected. Pre-position rescue assets."
        elif flood_prob >= 0.35:
            risk_level = "MODERATE"
            risk_color = "#eab308"  # Yellow
            alert_action = "ADVISORY WATCH: Saturated antecedent ground and moderate rainfall trends. Monitor regional gauge stations."
        else:
            risk_level = "LOW"
            risk_color = "#22c55e"  # Green
            alert_action = "NORMAL: Baseline conditions within seasonal thresholds. Routine monitoring ongoing."

        # Compute factor attributions
        total_precip_7d = env_feature_values.get("precip_7day_total_mm", 0.0)
        max_daily_precip = env_feature_values.get("precip_max_daily_mm", 0.0)
        last_1d_precip = env_feature_values.get("precip_last_1day_mm", 0.0)
        precip_trend = env_feature_values.get("precip_trend", 0.0)

        # Radar backscatter signal
        sat_np = satellite_tensor.cpu().numpy()[0]
        mean_vv = float(np.mean(sat_np[0]))
        mean_vh = float(np.mean(sat_np[1]))

        return {
            "region_name": region_name,
            "country": country,
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "forecast_date": forecast_date,
            "forecast_lead_time_days": lead_time_days,
            "flood_probability": round(flood_prob, 4),
            "risk_score_percent": round(flood_prob * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommended_action": alert_action,
            "key_factors": {
                "rainfall_7day_total_mm": round(total_precip_7d, 2),
                "peak_daily_rainfall_mm": round(max_daily_precip, 2),
                "recent_24h_rainfall_mm": round(last_1d_precip, 2),
                "precipitation_trend": "Increasing" if precip_trend > 0 else "Decreasing",
                "radar_mean_vv_scaled": round(mean_vv, 3),
                "radar_mean_vh_scaled": round(mean_vh, 3),
            },
        }


def get_default_monitored_regions() -> List[Dict[str, Any]]:
    """Default vulnerable monitored regions across India, Ghana, and Sri Lanka."""
    return [
        {
            "id": "ind-assam-brahmaputra",
            "name": "Brahmaputra Valley (Assam)",
            "country": "India",
            "latitude": 26.2006,
            "longitude": 92.9376,
            "basin": "Brahmaputra River Basin",
            "vulnerability_tier": "Very High",
        },
        {
            "id": "ind-bihar-kosi",
            "name": "Kosi River Basin (Bihar)",
            "country": "India",
            "latitude": 25.5941,
            "longitude": 85.1376,
            "basin": "Ganges-Kosi River Basin",
            "vulnerability_tier": "Very High",
        },
        {
            "id": "ind-kerala-periyar",
            "name": "Periyar Basin (Kerala)",
            "country": "India",
            "latitude": 10.0261,
            "longitude": 76.3125,
            "basin": "Periyar River Basin",
            "vulnerability_tier": "High",
        },
        {
            "id": "gha-volta-basin",
            "name": "Volta River Basin",
            "country": "Ghana",
            "latitude": 6.9337,
            "longitude": -1.1322,
            "basin": "Volta Basin",
            "vulnerability_tier": "High",
        },
        {
            "id": "lka-kelani-colombo",
            "name": "Kelani River Basin",
            "country": "Sri Lanka",
            "latitude": 6.9271,
            "longitude": 79.8612,
            "basin": "Kelani Ganga Basin",
            "vulnerability_tier": "High",
        },
    ]


if __name__ == "__main__":
    print("Testing NEXUS Regional Risk Engine...")
    engine = NexusRiskEngine()
    print("Risk Engine successfully initialized with trained Multimodal model!")

    # Test with sample input
    sample_sat = torch.zeros(2, 512, 512)
    sample_env = {"precip_7day_total_mm": 350.0, "precip_max_daily_mm": 120.0, "precip_last_1day_mm": 80.0, "precip_trend": 15.0}

    res = engine.assess_risk(
        region_name="Brahmaputra Valley (Assam)",
        country="India",
        latitude=26.2006,
        longitude=92.9376,
        forecast_date="2026-10-15",
        lead_time_days=3.0,
        satellite_tensor=sample_sat,
        env_feature_values=sample_env,
    )

    print()
    print("Sample Regional Assessment Output:")
    for k, v in res.items():
        print(f"  {k}: {v}")
