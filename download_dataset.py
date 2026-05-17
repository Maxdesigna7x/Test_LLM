#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_DATASET = "mattop/panda-or-bear-image-classification"


def ensure_kaggle_credentials() -> None:
    config_dir = Path.home() / ".config" / "kaggle"
    kaggle_json = config_dir / "kaggle.json"
    if kaggle_json.exists():
        return

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise SystemExit("Faltan KAGGLE_USERNAME/KAGGLE_KEY o ~/.config/kaggle/kaggle.json")

    config_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    kaggle_json.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga un dataset de Kaggle en dataset/")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Slug del dataset de Kaggle")
    parser.add_argument("--dest", default="dataset", help="Directorio destino")
    args = parser.parse_args()

    ensure_kaggle_credentials()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=str(dest), unzip=True, quiet=False)

    for archive in dest.glob("*.zip"):
        archive.unlink()

    print(f"Dataset descargado en: {dest.resolve()}")


if __name__ == "__main__":
    main()
