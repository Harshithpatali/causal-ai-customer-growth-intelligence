"""Command-line runner for repeated-split uplift robustness diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ai.data import load_data, validate_data
from causal_ai.robustness import repeated_split_robustness


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uplift ranking and policy value across repeated stratified splits."
    )
    parser.add_argument("--data", default="data/raw/hillstrom.csv")
    parser.add_argument("--output", default="reports/repeated_split_robustness.json")
    parser.add_argument("--seeds", default="11,23,37,51,73")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    frame = load_data(args.data)
    validation = validate_data(frame)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")

    result = repeated_split_robustness(
        frame,
        seeds=seeds,
        test_size=args.test_size,
        policy_threshold=args.threshold,
        artifact_path=args.output,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Detailed split results written to {Path(args.output)}")


if __name__ == "__main__":
    main()
