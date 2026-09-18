"""Entrypoint of the MLflow tracking server container.

MinIO starts empty, so before handing control to the actual command this
script makes sure the bucket from MLFLOW_ARTIFACTS_DESTINATION exists.
"""

import os
import sys
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ATTEMPTS = 30
DELAY_SECONDS = 2


def artifact_bucket() -> str | None:
    destination = urlparse(os.getenv("MLFLOW_ARTIFACTS_DESTINATION", ""))
    if destination.scheme != "s3" or not destination.netloc:
        return None
    return destination.netloc


def ensure_bucket(bucket: str) -> None:
    s3 = boto3.client("s3", endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"))
    for attempt in range(1, ATTEMPTS + 1):
        try:
            existing = {item["Name"] for item in s3.list_buckets().get("Buckets", [])}
            if bucket not in existing:
                s3.create_bucket(Bucket=bucket)
                print(f"Created artifact bucket s3://{bucket}", flush=True)
            return
        except (BotoCoreError, ClientError) as exc:
            print(f"Waiting for object storage ({attempt}/{ATTEMPTS}): {exc}", flush=True)
            time.sleep(DELAY_SECONDS)
    sys.exit(f"Object storage is unreachable, could not create bucket s3://{bucket}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: mlflow-entrypoint.py <command> [args...]")

    bucket = artifact_bucket()
    if bucket:
        ensure_bucket(bucket)

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
