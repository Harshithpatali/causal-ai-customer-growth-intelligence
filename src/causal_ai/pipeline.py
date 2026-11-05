import json
from pathlib import Path
from .data import load_data, validate_data
from .modeling import train_uplift

def run_pipeline(data_path="data/raw/hillstrom.csv", artifact_dir="artifacts"):
    df = load_data(data_path)
    validation = validate_data(df)
    if not validation["valid"]:
        raise ValueError(validation["issues"])
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    metrics = train_uplift(df, artifact_dir=artifact_dir)
    summary = {
        "validation": validation,
        "segment_counts": df["segment"].value_counts().to_dict(),
        "conversion_rate": float(df["conversion"].mean()),
        "visit_rate": float(df["visit"].mean()),
        "mean_spend": float(df["spend"].mean()),
        "treatment_conversion": float(df.loc[df.treatment==1, "conversion"].mean()),
        "control_conversion": float(df.loc[df.treatment==0, "conversion"].mean()),
        "treatment_spend": float(df.loc[df.treatment==1, "spend"].mean()),
        "control_spend": float(df.loc[df.treatment==0, "spend"].mean()),
        "metrics": metrics
    }
    Path(artifact_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
