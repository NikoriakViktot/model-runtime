import os
import boto3
from urllib.parse import urlparse

# 🔴 ЯВНО друкуємо env — щоб не було магії
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_DEFAULT_REGION:", os.getenv("AWS_DEFAULT_REGION"))
print("AWS_S3_BUCKET:", os.getenv("AWS_S3_BUCKET"))

s3 = boto3.client("s3")

# ⚠️ ВАЖЛИВО: bucket БЕЗ s3://
BUCKET = os.getenv("AWS_S3_BUCKET")

# 🔴 ВСТАВ СЮДИ РЕАЛЬНИЙ key З CONTROL PLANE
KEY = "datasets/Lisa Moore/af372a33-aeb5-4e75-b6a2-e7f28f0468ef/graph.jsonl"

print("Trying to download:")
print("  bucket =", BUCKET)
print("  key    =", KEY)

try:
    s3.head_object(Bucket=BUCKET, Key=KEY)
    print("✅ S3 object exists")

    s3.download_file(BUCKET, KEY, "/tmp/test.jsonl")
    print("✅ Downloaded to /tmp/test.jsonl")

    print("\n--- PREVIEW ---")
    with open("/tmp/test.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i == 2:
                break

except Exception as e:
    print("❌ S3 ERROR:")
    print(type(e).__name__, e)
