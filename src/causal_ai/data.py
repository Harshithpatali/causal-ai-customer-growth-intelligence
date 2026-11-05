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
    """Validate the experiment schema and core outcome invariants before analysis."""
    issues = []
    missing_cols = sorted(set(REQUIRED) - set(df.columns))
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    if df.empty:
        issues.append("Dataset is empty")
        return {"valid": False, "issues": issues, "rows": 0,
                "columns": int(df.shape[1]), "duplicate_rows": 0,
                "missing_cells": int(df.isna().sum().sum())}

    present_required = [column for column in REQUIRED if column in df.columns]
    missing_required = int(df[present_required].isna().sum().sum())
    if missing_required:
        issues.append(f"Required columns contain {missing_required} missing cells")

    for column in ["visit", "conversion", "mens", "womens", "newbie"]:
        if column in df and not set(df[column].dropna().unique()).issubset({0, 1}):
            issues.append(f"{column} must contain only binary values (0/1)")

    for column in ["recency", "history", "spend"]:
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.isna().any() or not values.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
                issues.append(f"{column} must contain finite numeric values")

    if "recency" in df and pd.to_numeric(df["recency"], errors="coerce").dropna().lt(0).any():
        issues.append("recency must be non-negative")
    if "history" in df and pd.to_numeric(df["history"], errors="coerce").dropna().lt(0).any():
        issues.append("history must be non-negative")
    if "spend" in df and pd.to_numeric(df["spend"], errors="coerce").dropna().lt(0).any():
        issues.append("spend must be non-negative")

    if "segment" in df:
        allowed = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
        unexpected = set(df["segment"].dropna().unique()) - allowed
        if unexpected:
            issues.append(f"Unexpected segments: {sorted(unexpected)}")

    if {"conversion", "spend"}.issubset(df.columns):
        conversion = pd.to_numeric(df["conversion"], errors="coerce")
        spend = pd.to_numeric(df["spend"], errors="coerce")
        consistent = conversion.eq((spend > 0).astype(int))
        if not consistent.all():
            issues.append("conversion must equal 1 when spend is positive and 0 otherwise")

    return {
        "valid": not issues,
        "issues": issues,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }
