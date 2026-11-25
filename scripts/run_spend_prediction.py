from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib

from causal_ai.data import load_data, validate_data
from causal_ai.forecasting import run_spend_prediction


def main():
    parser = argparse.ArgumentParser(description="Run honest Hillstrom spend prediction evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", default="reports/spend_prediction.json")
    parser.add_argument("--model-output", default="artifacts/spend_forecaster.joblib")
    args = parser.parse_args()

    df = load_data()
    validation = validate_data(df)
    if not validation["valid"]:
        raise SystemExit(f"Dataset validation failed: {validation['issues']}")

    report, model = run_spend_prediction(df, seed=args.seed, alpha=args.alpha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(json.dumps({
        "report": str(output),
        "model": str(model_path),
        "baseline_mae": report["baseline"]["mae"],
        "model_mae": report["model_metrics"]["mae"],
        "model_rmse": report["model_metrics"]["rmse"],
        "model_r2": report["model_metrics"]["r2"],
        "conformal_coverage": report["model_metrics"]["conformal_interval"]["empirical_coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
