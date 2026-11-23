"""Run reproducible uplift learner benchmarks on Hillstrom data."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from causal_ai.data import load_data, validate_data
from causal_ai.uplift_benchmark import (
    _fit_predict_fold,
    repeated_crossfit_benchmark,
    repeated_multitreatment_benchmark,
)
from causal_ai.research_diagnostics import (
    assignment_probability_stress_test,
    ranking_metrics,
    randomized_design_diagnostics,
    subgroup_effects,
)


def _ranked_diagnostics(df, outcome, features, seeds, folds, propensity):
    """Generate OOF rankings independently of the policy-value benchmark."""
    t = df["treatment"].astype(int).to_numpy()
    y = df[outcome].astype(float).to_numpy()
    X = df[features]
    rows = []
    elapsed = {}
    for seed in seeds:
        started = time.perf_counter()
        scores = {name: np.zeros(len(df)) for name in
                  ["s_learner", "t_learner", "x_learner", "dr_learner"]}
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        for train, valid in splitter.split(X, t):
            pred = _fit_predict_fold(X.iloc[train], t[train], y[train],
                                     X.iloc[valid], propensity, int(seed))
            for name, values in pred.items():
                scores[name][valid] = values
        rules = {
            "random": np.random.default_rng(int(seed)).random(len(df)),
            "recency": -df["recency"].to_numpy(dtype=float),
            "historical_spend": df["history"].to_numpy(dtype=float),
            "mens_affinity": df["mens"].to_numpy(dtype=float),
            "womens_affinity": df["womens"].to_numpy(dtype=float),
        }
        scores.update(rules)
        for name, score in scores.items():
            rows.append({"seed": int(seed), "ranker": name,
                         **ranking_metrics(y, t, score, propensity)})
        elapsed[str(seed)] = time.perf_counter() - started
    # Mean and split-to-split SD are descriptive, not independent-experiment CIs.
    aggregate = {}
    for ranker in sorted({r["ranker"] for r in rows}):
        subset = [r for r in rows if r["ranker"] == ranker]
        aggregate[ranker] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in subset])),
                "sd_across_seeds": float(np.std([r[metric] for r in subset], ddof=1))
                if len(subset) > 1 else None,
            }
            for metric in ["auuc", "qini_area", "overall_ate_ipw"]
        }
        for fraction in ["10%", "20%", "30%"]:
            vals = [r["uplift_at"][fraction]["estimate"] for r in subset]
            aggregate[ranker].setdefault("uplift_at", {})[fraction] = {
                "mean": float(np.mean(vals)),
                "sd_across_seeds": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            }
    return {"by_seed": rows, "aggregate": aggregate,
            "runtime_seconds_by_seed": elapsed,
            "runtime_seconds_total": float(sum(elapsed.values())),
            "uncertainty_note": "Across-seed SD measures split sensitivity; it is not a sampling confidence interval."}


def _policy_comparison(binary_result):
    summary = {row["policy"]: row for row in binary_result["summary"]}
    learned = ["s_learner", "t_learner", "x_learner", "dr_learner"]
    baselines = ["random", "recency", "historical_spend",
                 "mens_affinity", "womens_affinity", "all_treat", "none"]
    rows = []
    for model in learned:
        for baseline in baselines:
            a, b = summary[model], summary[baseline]
            rows.append({
                "model_policy": model, "baseline_policy": baseline,
                "model_value": a["mean_value"], "baseline_value": b["mean_value"],
                "incremental_value_per_customer": a["mean_value"] - b["mean_value"],
                "conservative_ci_lower": a["mean_ci_lower"] - b["mean_ci_upper"],
                "conservative_ci_upper": a["mean_ci_upper"] - b["mean_ci_lower"],
                "interval_note": "Conservative difference of marginal 95% IPW intervals; not a paired interval.",
            })
    return rows


def main():
    started = time.perf_counter()
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
    if not seeds or args.folds < 2:
        raise ValueError("At least one seed and two or more folds are required")
    features = ["recency", "history_segment", "history", "mens", "womens",
                "zip_code", "newbie", "channel"]
    binary = repeated_crossfit_benchmark(
        df, outcome_col=args.outcome, feature_cols=features,
        seeds=seeds, n_splits=args.folds, propensity=2/3)
    creative = repeated_multitreatment_benchmark(
        df, outcome_col=args.outcome, feature_cols=features,
        seeds=seeds, n_splits=args.folds)
    ranking = _ranked_diagnostics(df, args.outcome, features, seeds, args.folds, 2/3)
    design = randomized_design_diagnostics(df, propensity=2/3)
    design["subgroups"] = subgroup_effects(df, args.outcome, propensity=2/3)
    design["assignment_probability_stress_test"] = assignment_probability_stress_test(
        df, args.outcome, known_propensity=2/3)
    payload = {
        "dataset": "Hillstrom / MineThatData randomized email experiment",
        "rows": len(df), "outcome": args.outcome, "seeds": list(seeds),
        "folds": args.folds, "binary_pooled_email_vs_control": binary,
        "creative_specific_vs_control": creative,
        "ranking_metrics": ranking,
        "randomized_design_diagnostics": design,
        "incremental_policy_value_vs_baselines": _policy_comparison(binary),
        "runtime_seconds_total": float(time.perf_counter() - started),
        "interpretation": (
            "Cross-fitted policy values use known randomization propensities. "
            "No observational propensity model is fitted. Qini/AUUC and uplift-at-k "
            "use the known-propensity IPW pseudo-outcome and are ranking diagnostics, "
            "not profit estimates. Across-seed SD measures split sensitivity, not "
            "independent-sample uncertainty. Conservative policy-difference intervals "
            "are differences of marginal IPW intervals, not paired intervals. "
            "Creative-specific contrasts are separate randomized comparisons, not "
            "a joint three-action policy or direct creative superiority test."
        ),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved benchmark report: {path}")
    print(f"Elapsed seconds: {payload['runtime_seconds_total']:.2f}")


if __old_sha_missing__:
    pass
