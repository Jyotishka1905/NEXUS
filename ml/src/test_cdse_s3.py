import os
import boto3
from botocore.config import Config

ACCESS_KEY = os.environ["CDSE_S3_ACCESS_KEY"]
SECRET_KEY = os.environ["CDSE_S3_SECRET_KEY"]

BUCKET = "eodata"

PREFIX = (
    "Sentinel-1/SAR/IW_GRDH_1S-COG/"
    "2016/03/09/"
    "S1A_IW_GRDH_1SSV_20160309T114019_20160309T114044_010292_"
    "00F39B_529A_COG.SAFE/"
)

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="default",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)

print("Testing CDSE S3 access...")
print()

# ---------------------------------------------------------
# 1. Test listing the product prefix
# ---------------------------------------------------------

print("Testing product prefix...")

response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix=PREFIX,
    MaxKeys=10,
)

print("List request succeeded.")
print("Objects returned:", response.get("KeyCount", 0))
print()

for obj in response.get("Contents", []):
    print(
        obj["Key"],
        "|",
        obj["Size"],
        "bytes"
    )

print()

# ---------------------------------------------------------
# 2. Find the measurement TIFF
# ---------------------------------------------------------

objects = response.get("Contents", [])

tiff_objects = [
    obj
    for obj in objects
    if obj["Key"].lower().endswith(".tiff")
]

if not tiff_objects:
    raise RuntimeError(
        "No TIFF found in the first 10 objects."
    )

tiff_key = tiff_objects[0]["Key"]

print("Found TIFF:")
print(tiff_key)
print()

# ---------------------------------------------------------
# 3. Test HEAD
# ---------------------------------------------------------

print("Testing HEAD request...")

head = s3.head_object(
    Bucket=BUCKET,
    Key=tiff_key,
)

print("HEAD succeeded.")
print("Size:", head["ContentLength"], "bytes")
print("Content type:", head.get("ContentType"))
print()

# ---------------------------------------------------------
# 4. Download ONLY first 1024 bytes
# ---------------------------------------------------------

print("Testing byte-range request...")

response = s3.get_object(
    Bucket=BUCKET,
    Key=tiff_key,
    Range="bytes=0-1023",
)

data = response["Body"].read()

print()
print("Range request succeeded.")
print("HTTP status:", response["ResponseMetadata"]["HTTPStatusCode"])
print("Bytes received:", len(data))
print("Content range:", response.get("ContentRange"))