"""Run reproducible uplift learner benchmarks on Hillstrom data."""
import argparse
import json
from pathlib import Path

from causal_ai.data import load_data, validate_data
from causal_ai.uplift_benchmark import (
    repeated_crossfit_benchmark,
    repeated_multitreatment_benchmark,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/hillstrom.csv")
    parser.add_argument("--out", default="reports/uplift_benchmark.json")
    parser.add_argument("--outcome", default="conversion",
                        choices=["conversion", "visit", "spend"])
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    df = load_data(args.data)
    validation = validate_data(df)
    if not validation["valid"]:
        raise ValueError(validation["issues"])
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    features = ["recency", "history_segment", "history", "mens", "womens",
                "zip_code", "newbie", "channel"]
    binary = repeated_crossfit_benchmark(
        df, outcome_col=args.outcome, feature_cols=features,
        seeds=seeds, n_splits=args.folds, propensity=2/3)
    creative = repeated_multitreatment_benchmark(
        df, outcome_col=args.outcome, feature_cols=features,
        seeds=seeds, n_splits=args.folds)
    payload = {
        "dataset": "Hillstrom / MineThatData randomized email experiment",
        "rows": len(df), "outcome": args.outcome, "seeds": list(seeds),
        "folds": args.folds, "binary_pooled_email_vs_control": binary,
        "creative_specific_vs_control": creative,
        "interpretation": (
            "Cross-fitted policy values use known randomization propensities. "
            "The mean and SD across seeds describe split sensitivity, not "
            "independent-sample confidence intervals. Creative-specific "
            "contrasts are separate randomized comparisons, not a joint "
            "three-action policy or direct creative superiority test."
        ),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved benchmark report: {path}")


if __name__ == "__main__":
    main()
