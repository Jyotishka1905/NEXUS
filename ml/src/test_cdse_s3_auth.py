import os
import boto3

access_key = os.environ["CDSE_S3_ACCESS_KEY"]
secret_key = os.environ["CDSE_S3_SECRET_KEY"]

endpoints = [
    "https://eodata.dataspace.copernicus.eu",
    "https://eodata.ams.dataspace.copernicus.eu",
]

for endpoint in endpoints:
    print("=" * 70)
    print("Testing:", endpoint)

    try:
        s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="default",
        )

        bucket = s3.Bucket("eodata")

        # Use the same style as CDSE's official boto3 example.
        objects = bucket.objects.filter(
            Prefix="Sentinel-1/SAR/IW_GRDH_1S-COG/2016/03/09/"
        )

        count = 0

        for obj in objects:
            print("FOUND:", obj.key)
            print("SIZE:", obj.size)
            count += 1

            if count >= 3:
                break

        print("SUCCESS - objects found:", count)

    except Exception as e:
        print("FAILED")
        print(type(e).__name__)
        print(e)

    print()