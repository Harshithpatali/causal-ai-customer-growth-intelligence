"""Run the meta-learner, ranking-baseline and creative-level benchmark."""
from __future__ import annotations

import argparse
import json

from causal_ai.data import load_data, validate_data
from causal_ai.benchmarking import run_uplift_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark uplift meta-learners, targeting baselines and email creatives."
    )
    parser.add_argument("--data", default="data/raw/hillstrom.csv")
    parser.add_argument("--output", default="reports/uplift_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    frame = load_data(args.data)
    validation = validate_data(frame)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")
    result = run_uplift_benchmark(
        frame, test_size=args.test_size, seed=args.seed,
        n_bootstrap=args.bootstrap, artifact_path=args.output,
    )
    print(json.dumps({
        "design": result["design"],
        "binary_uplift_benchmark": result["binary_uplift_benchmark"],
        "bootstrap_policy_contrasts": result["bootstrap_policy_contrasts"],
        "multi_treatment_average_effects": result["multi_treatment_average_effects"],
        "multi_treatment_targeting": result["multi_treatment_targeting"],
    }, indent=2))
    print(f"Full benchmark written to {args.output}")


if __name__ == "__main__":
    main()
