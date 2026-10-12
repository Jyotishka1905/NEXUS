import os
from pathlib import Path
import configparser
import pandas as pd
import numpy as np

try:
    import boto3
    from botocore.config import Config
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSET_FILE = PROJECT_ROOT / "data" / "environmental" / "processed" / "satellite_forecast_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "forecast_satellite_chips"


def get_s3_credentials():
    # Check env vars first
    access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
    secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
    if access_key and secret_key:
        return access_key, secret_key

    # Check ~/.s3cfg
    s3cfg_path = Path.home() / ".s3cfg"
    if s3cfg_path.exists():
        parser = configparser.ConfigParser()
        parser.read(s3cfg_path)
        if "default" in parser:
            access_key = parser["default"].get("access_key")
            secret_key = parser["default"].get("secret_key")
            if access_key and secret_key:
                return access_key, secret_key

    return None, None


def main():
    print("NEXUS CDSE S3 Sentinel-1 Downloader & Cropper")
    print("============================================")

    access_key, secret_key = get_s3_credentials()
    if not access_key or not secret_key:
        print("[!] No CDSE S3 credentials found in environment or ~/.s3cfg.")
        print("    Configure credentials with 's3cmd --configure' or set CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY.")
        return

    print("Found CDSE S3 credentials. Connecting to eodata.dataspace.copernicus.eu...")
    s3 = boto3.client(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    df = pd.read_csv(SUBSET_FILE)
    print(f"Loaded {len(df)} samples from {SUBSET_FILE.name}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # List or download measurement TIFFs for each product
    print("Ready to process Sentinel-1 measurement TIFFs.")


if __name__ == "__main__":
    main()
