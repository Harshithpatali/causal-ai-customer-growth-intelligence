"""Cross-fitted AIPW, CUPED, balance, multiplicity and power analysis."""
import argparse
import json
from pathlib import Path

import numpy as np

from causal_ai.data import FEATURES, load_data, validate_data
from causal_ai.advanced_statistics import cross_fitted_spend_predictions
from causal_ai.statistics import (
    aipw_average_treatment_effect, cuped_adjusted_ate, holm_adjust,
    randomized_assignment_diagnostics, two_proportion_sample_size,
    two_proportion_test, mean_difference,
)


def run_advanced_statistics(df, seed=42, folds=5, mde=0.005):
    validation = validate_data(df)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")
    t = df["treatment"].astype(int).to_numpy()
    y_spend = df["spend"].astype(float).to_numpy()
    results = {}
    tests = {
        "visit": two_proportion_test(df["visit"], t),
        "conversion": two_proportion_test(df["conversion"], t),
        "spend": mean_difference(y_spend, t),
    }
    pvalues = [tests[name]["p_value"] for name in ("visit", "conversion", "spend")]
    adjusted = holm_adjust(pvalues)
    for name, p_adjusted in zip(("visit", "conversion", "spend"), adjusted):
        results[name] = {**tests[name], "holm_adjusted_p_value": p_adjusted}

    results["spend_cuped"] = cuped_adjusted_ate(y_spend, t, df["history"].astype(float))
    mu1, mu0 = cross_fitted_spend_predictions(df, seed=seed, folds=folds)
    results["spend_aipw"] = aipw_average_treatment_effect(
        y_spend, t, 2/3, mu1, mu0
    )
    results["randomization_balance"] = randomized_assignment_diagnostics(
        df, treatment_col="treatment", features=FEATURES
    )
    control_rate = float(df.loc[df["treatment"] == 0, "conversion"].mean())
    treatment_rate = min(0.99, control_rate + mde)
    results["conversion_power_plan"] = two_proportion_sample_size(
        control_rate, treatment_rate, alpha=0.05, power=0.80
    )
    results["design"] = {
        "n": int(len(df)), "crossfit_folds": int(folds), "seed": int(seed),
        "pooled_email_propensity": 2/3,
        "holm_family": ["visit", "conversion", "spend"],
        "power_plan_mde": float(mde),
        "interpretation": "AIPW nuisance predictions are cross-fitted; CUPED uses pre-treatment historical spend. Power calculation is a planning approximation."
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mde", type=float, default=0.005)
    parser.add_argument("--output", default="reports/advanced_statistics.json")
    args = parser.parse_args()
    result = run_advanced_statistics(load_data(), args.seed, args.folds, args.mde)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote advanced statistical analysis: {output}")


if __name__ == "__main__":
    main()
