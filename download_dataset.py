#!/usr/bin/env python3
"""Download and unpack the Kaggle cat vs dog dataset."""

from __future__ import annotations

import base64
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_SLUG = "tongpython/cat-and-dog"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"


def main() -> int:
    root = Path(__file__).resolve().parent
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if (dataset_dir / "PetImages").exists():
        print(f"Dataset already present at {dataset_dir}")
        return 0

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("Set KAGGLE_USERNAME and KAGGLE_KEY before running.", file=sys.stderr)
        return 1

    zip_path = dataset_dir / "cat-and-dog.zip"
    auth = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={
            "Authorization": f"Basic {auth}",
            "Accept": "application/octet-stream",
            "User-Agent": "Mozilla/5.0",
        },
    )

    print(f"Downloading {DATASET_SLUG} to {zip_path}")
    try:
        with urllib.request.urlopen(request) as response, zip_path.open("wb") as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
    except urllib.error.HTTPError as exc:
        print(f"Download failed: {exc.code} {exc.reason}", file=sys.stderr)
        return 1

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)

    zip_path.unlink(missing_ok=True)
    print(f"Dataset ready at {dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
