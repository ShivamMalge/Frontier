# src/frontier/utils/helpers.py

import os


def ensure_dirs():
    for path in ["data", "data/raw", "data/processed", "data/models"]:
        os.makedirs(path, exist_ok=True)
