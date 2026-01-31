# src/utils.py
import os
import joblib
import pandas as pd
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, name):
    path = ARTIFACTS / name
    joblib.dump(obj, path)
    return path

def load_pickle(name):
    path = ARTIFACTS / name
    return joblib.load(path)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_read_csv(path, parse_dates=["Date"]):
    df = pd.read_csv(path, parse_dates=parse_dates)
    # normalize columns
    df = df.rename(columns=lambda c: c.strip())
    return df
