"""Repeated-seed uplift benchmark, diagnostics and interactive research charts.

The five splits reuse the same historical randomized experiment. Cross-seed
variation is a sensitivity diagnostic, not independent replication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from causal_ai.data import load_data, validate_data
from causal_ai.benchmarking import run_uplift_benchmark


DEFAULT_SEEDS = (21, 42, 63, 84, 105)


def _metric_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "values_by_seed": values.tolist(),
    }


def _aggregate(results, seeds):
    names = list(results[0]["binary_uplift_benchmark"])
    metrics = {}
    for name in names:
        metrics[name] = {
            "policy_value_ipw": _metric_summary([
                run["binary_uplift_benchmark"][name]["policy_value_ipw"]["value"]
                for run in results
            ]),
            "auuc_ht": _metric_summary([
                run["binary_uplift_benchmark"][name]["auuc_ht"] for run in results
            ]),
            "qini_ht": _metric_summary([
                run["binary_uplift_benchmark"][name]["qini_ht"] for run in results
            ]),
        }

    winner_counts = {
        "policy_value": {
            name: sum(
                max(names, key=lambda n: run["binary_uplift_benchmark"][n]["policy_value_ipw"]["value"]) == name
                for run in results
            )
            for name in names
        },
        "auuc": {
            name: sum(
                max(names, key=lambda n: run["binary_uplift_benchmark"][n]["auuc_ht"]) == name
                for run in results
            )
            for name in names
        },
        "qini": {
            name: sum(
                max(names, key=lambda n: run["binary_uplift_benchmark"][n]["qini_ht"]) == name
                for run in results
            )
            for name in names
        },
    }

    feature_names = list(results[0]["model_feature_importance"]["S-learner"])
    feature_importance = {
        learner: {
            feature: float(np.mean([
                run["model_feature_importance"][learner].get(feature, 0.0)
                for run in results
            ]))
            for feature in feature_names
        }
        for learner in ("T-learner", "S-learner", "X-learner")
    }
    score_quantiles = {
        learner: {
            quantile: _metric_summary([
                run["uplift_score_diagnostics"][learner]["quantiles"][quantile]
                for run in results
            ])
            for quantile in ("p10", "p25", "p50", "p75", "p90")
        }
        for learner in ("T-learner", "S-learner", "X-learner")
    }
    return {
        "seeds": list(seeds),
        "n_splits": len(seeds),
        "interpretation": (
            "Repeated stratified partitions of the same 64,000-row experiment. "
            "Cross-seed standard deviations describe split sensitivity, not "
            "independent experimental replication or external validation."
        ),
        "metric_summary": metrics,
        "winner_counts": winner_counts,
        "mean_feature_importance": feature_importance,
        "uplift_score_quantile_stability": score_quantiles,
    }


def _build_figure(results, seeds, summary, output):
    names = list(results[0]["binary_uplift_benchmark"])
    learners = ("T-learner", "S-learner", "X-learner")
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Holdout policy value across seeds",
            "Cumulative HT uplift gain (mean ± 1 SD across seeds)",
            "AUUC across seeds",
            "Qini across seeds",
            "Feature importance and model interpretation",
            "Mean ExtraTrees feature importance",
            "Predicted uplift-score quantiles across seeds",
            "HT pseudo-outcome by predicted-uplift decile",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )

    for name in names:
        vals = [r["binary_uplift_benchmark"][name]["policy_value_ipw"]["value"] for r in results]
        fig.add_trace(go.Box(name=name, y=vals, boxpoints="all", jitter=0.25), row=1, col=1)

    grid = np.linspace(0, 1, 101)
    for name in names:
        curves = []
        for run in results:
            curve = run["binary_uplift_benchmark"][name]["cumulative_curve"]
            curves.append(np.interp(grid, curve["fraction"], curve["gain"]))
        curves = np.asarray(curves)
        mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
        fig.add_trace(go.Scatter(x=grid, y=mean + sd, mode="lines",
                                 line={"width": 0}, showlegend=False,
                                 hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=grid, y=mean - sd, mode="lines",
                                 line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(100,100,100,0.08)",
                                 showlegend=False, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=grid, y=mean, mode="lines", name=name,
                                 legendgroup=name), row=1, col=2)
    fig.add_trace(go.Scatter(x=grid, y=np.zeros_like(grid), mode="lines",
                             name="Zero gain", line={"dash": "dash"}), row=1, col=2)

    for name in names:
        vals = [r["binary_uplift_benchmark"][name]["auuc_ht"] for r in results]
        fig.add_trace(go.Box(name=name, y=vals, boxpoints="all", showlegend=False), row=2, col=1)
        vals = [r["binary_uplift_benchmark"][name]["qini_ht"] for r in results]
        fig.add_trace(go.Box(name=name, y=vals, boxpoints="all", showlegend=False), row=2, col=2)

    seed_index = list(seeds).index(42) if 42 in seeds else 0
    primary = results[seed_index]
    for learner in learners:
        contrast = primary["bootstrap_policy_contrasts_vs_email_everyone"][learner + " vs Email everyone"]
        center = contrast["estimate"]
        fig.add_trace(go.Scatter(
            x=[learner], y=[center], mode="markers", name=learner,
            error_y={"type": "data", "symmetric": False,
                     "array": [contrast["bootstrap_ci_upper"] - center],
                     "arrayminus": [center - contrast["bootstrap_ci_lower"]]},
            showlegend=False,
        ), row=4, col=2)

    for learner in learners:
        items = summary["mean_feature_importance"][learner]
        ordered = sorted(items.items(), key=lambda pair: pair[1], reverse=True)
        fig.add_trace(go.Bar(
            x=[v for _, v in ordered], y=[k for k, _ in ordered],
            orientation="h", name=learner,
        ), row=3, col=1)

    for learner in learners:
        q = summary["uplift_score_quantile_stability"][learner]
        x = ["p10", "p25", "p50", "p75", "p90"]
        means = [q[k]["mean"] for k in x]
        errors = [q[k]["standard_deviation"] for k in x]
        fig.add_trace(go.Scatter(
            x=x, y=means, mode="lines+markers", name=learner,
            error_y={"type": "data", "array": errors},
        ), row=3, col=2)

    for learner in learners:
        decile_means = []
        for d in range(10):
            decile_means.append(np.mean([
                run["uplift_score_heterogeneity_deciles"][learner][d]["mean_ht_pseudo_outcome"]
                for run in results
            ]))
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=decile_means,
            mode="lines+markers", name=learner,
        ), row=4, col=1)

    fig.update_xaxes(title_text="Fraction targeted", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative HT gain", row=1, col=2)
    fig.update_yaxes(title_text="IPW spend value per customer", row=1, col=1)
    fig.update_yaxes(title_text="AUUC", row=2, col=1)
    fig.update_yaxes(title_text="Qini", row=2, col=2)
    fig.update_xaxes(title_text="Mean impurity importance", row=3, col=1)
    fig.update_xaxes(title_text="Predicted uplift quantile", row=3, col=2)
    fig.update_yaxes(title_text="Expected-spend uplift score", row=3, col=2)
    fig.update_xaxes(title_text="Predicted-uplift decile (low to high)", row=4, col=1)
    fig.update_yaxes(title_text="Mean HT pseudo-outcome", row=4, col=1)
    fig.update_yaxes(title_text="Targeted minus email-everyone value", row=4, col=2)
    fig.update_layout(
        title="Hillstrom uplift benchmark — repeated-seed diagnostics",
        height=1900, width=1500, boxmode="group",
        template="plotly_white",
    )
    fig.write_html(output, include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated-seed uplift benchmarking and publish diagnostic charts."
    )
    parser.add_argument("--data", default="data/raw/hillstrom.csv")
    parser.add_argument("--output", default="reports/repeated_seed_benchmark.json")
    parser.add_argument("--chart", default="reports/repeated_seed_diagnostics.html")
    parser.add_argument("--seeds", default="21,42,63,84,105")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=100,
                        help="Paired bootstrap replicates per split (minimum 100).")
    args = parser.parse_args()

    seeds = tuple(int(v.strip()) for v in args.seeds.split(",") if v.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more unique comma-separated seeds.")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100.")

    frame = load_data(args.data)
    validation = validate_data(frame)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")

    results = []
    for seed in seeds:
        print(f"Running benchmark seed={seed}, bootstrap={args.bootstrap}", flush=True)
        results.append(run_uplift_benchmark(
            frame, test_size=args.test_size, seed=seed,
            n_bootstrap=args.bootstrap,
        ))

    summary = _aggregate(results, seeds)
    report = {
        "summary": summary,
        "per_seed_results": results,
        "methodology": {
            "split": "Stratified train/holdout split, repeated over the requested seeds.",
            "primary_seeds": list(seeds),
            "bootstrap_per_split": args.bootstrap,
            "cross_seed_uncertainty": "Descriptive split sensitivity on the same dataset; not independent replications.",
            "feature_importance": "Impurity importance from ExtraTrees outcome/effect models; predictive rather than causal.",
            "heterogeneity": "Predicted-score deciles summarized using holdout HT pseudo-outcomes; exploratory and noisy.",
            "policy_value": "Holdout Horvitz-Thompson IPW, fixed policy budget equal to pooled randomized email share.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    chart = Path(args.chart)
    chart.parent.mkdir(parents=True, exist_ok=True)
    _build_figure(results, seeds, summary, chart)
    print(json.dumps(summary, indent=2))
    print(f"Report written to {output}")
    print(f"Interactive diagnostics written to {chart}")


if __name__ == "__main__":
    main()
