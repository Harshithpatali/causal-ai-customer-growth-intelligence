from pathlib import Path
import pandas as pd
import requests

DATA_URLS = [
    "https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv",
    "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
]
REQUIRED = ["recency","history_segment","history","mens","womens","zip_code",
            "newbie","channel","segment","visit","conversion","spend"]
FEATURES = ["recency","history_segment","history","mens","womens",
            "zip_code","newbie","channel"]

def download_dataset(path="data/raw/hillstrom.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    errors = []
    for url in DATA_URLS:
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            if len(response.content) < 100_000 or b"recency" not in response.content[:500]:
                raise ValueError("Downloaded content does not look like the Hillstrom CSV")
            path.write_bytes(response.content)
            return path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download Hillstrom dataset. Download it manually from "
                       "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html "
                       f"and save it to {path}. Details: {' | '.join(errors)}")

def load_data(path="data/raw/hillstrom.csv", download=True):
    p = Path(path)
    if not p.exists() and download:
        download_dataset(p)
    df = pd.read_csv(p)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.copy()
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    df["treatment_label"] = df["treatment"].map({0:"Control", 1:"Email treatment"})
    for c in ["mens", "womens", "newbie", "visit", "conversion"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    df["spend"] = pd.to_numeric(df["spend"], errors="raise")
    return df

def validate_data(df):
    issues = []
    missing_cols = sorted(set(REQUIRED) - set(df.columns))
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    if df.empty:
        issues.append("Dataset is empty")
    for c in ["visit", "conversion", "mens", "womens", "newbie"]:
        if c in df and not set(df[c].dropna().unique()).issubset({0,1}):
            issues.append(f"{c} must be binary")
    if "segment" in df:
        allowed = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
        unexpected = set(df["segment"].dropna().unique()) - allowed
        if unexpected:
            issues.append(f"Unexpected segments: {sorted(unexpected)}")
    if "spend" in df and (df["spend"].dropna() < 0).any():
        issues.append("Spend contains negative values")
    return {"valid": not issues, "issues": issues,
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum())}
