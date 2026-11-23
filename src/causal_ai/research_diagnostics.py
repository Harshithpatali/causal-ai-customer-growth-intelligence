"""Ranking and randomized-design diagnostics for uplift research benchmarks.

The evaluation score is the Horvitz-Thompson pseudo-outcome using the
experiment's known assignment probability. No observational propensity model
is fitted. Qini/AUUC are descriptive ranking metrics, not profit estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def randomized_pseudo_outcome(y, treatment, propensity):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    p = float(propensity)
    if y.ndim != 1 or t.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("y and treatment must be aligned finite vectors")
    if not set(np.unique(t)).issubset({0, 1}) or len(np.unique(t)) != 2:
        raise ValueError("both randomized treatment arms are required")
    if not 0 < p < 1:
        raise ValueError("known assignment propensity must be in (0, 1)")
    return t * y / p - (1 - t) * y / (1 - p)


def ranking_metrics(y, treatment, score, propensity, fractions=(0.10, 0.20, 0.30)):
    """Compute AUUC, Qini area and top-k uplift from known-propensity scores."""
    score = np.asarray(score, dtype=float)
    tau = randomized_pseudo_outcome(y, treatment, propensity)
    if score.shape != tau.shape or not np.isfinite(score).all():
        raise ValueError("score must be finite and aligned with outcomes")
    order = np.argsort(-score, kind="stable")
    ranked = tau[order]
    n = len(ranked)
    x = np.arange(1, n + 1, dtype=float) / n
    cumulative_uplift = np.cumsum(ranked) / np.arange(1, n + 1)
    overall = float(np.mean(tau))
    qini_curve = np.cumsum(ranked) / n - x * overall
    auuc = float(np.trapz(np.r_[0.0, cumulative_uplift], np.r_[0.0, x]))
    qini_area = float(np.trapz(np.r_[0.0, qini_curve], np.r_[0.0, x]))
    uplift_at = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("fractions must be in (0, 1]")
        k = max(1, int(np.ceil(fraction * n)))
        selected = ranked[:k]
        estimate = float(np.mean(selected))
        se = float(np.std(selected, ddof=1) / np.sqrt(k)) if k > 1 else None
        uplift_at[f"{int(round(fraction * 100))}%"] = {
            "estimate": estimate,
            "ci_lower": estimate - 1.96 * se if se is not None else None,
            "ci_upper": estimate + 1.96 * se if se is not None else None,
            "n_selected": int(k),
        }
    return {
        "auuc": auuc,
        "qini_area": qini_area,
        "overall_ate_ipw": overall,
        "uplift_at": uplift_at,
        "metric_note": "IPW pseudo-outcome ranking metrics; confidence intervals at k are conditional normal approximations",
    }


def randomized_design_diagnostics(df, treatment_col="treatment", propensity=2 / 3,
                                  subgroup_cols=("history_segment", "mens", "womens",
                                                 "newbie", "channel")):
    """Check known randomized allocation and report descriptive subgroup effects."""
    if treatment_col not in df:
        raise ValueError(f"Missing treatment column: {treatment_col}")
    t = df[treatment_col].astype(int).to_numpy()
    if not set(np.unique(t)).issubset({0, 1}) or len(np.unique(t)) != 2:
        raise ValueError("both treatment arms must be present")
    p = float(propensity)
    counts = pd.Series(t).value_counts().to_dict()
    observed = float(np.mean(t))
    result = {
        "propensity_source": "known randomized assignment design",
        "known_treatment_probability": p,
        "observed_treatment_share": observed,
        "observed_control_share": 1.0 - observed,
        "treatment_count": int(counts.get(1, 0)),
        "control_count": int(counts.get(0, 0)),
        "design_overlap": {
            "treatment_probability": p,
            "control_probability": 1.0 - p,
            "minimum_arm_probability": min(p, 1.0 - p),
            "maximum_arm_probability": max(p, 1.0 - p),
            "positivity_satisfied_by_design": bool(0 < p < 1),
            "individual_propensity_model_fitted": False,
        },
        "allocation_deviation": observed - p,
        "subgroups": {},
    }
    # The binary pooled contrast uses the randomized IPW pseudo-outcome.
    return result


def subgroup_effects(df, outcome_col, treatment_col="treatment", propensity=2 / 3,
                     subgroup_cols=("history_segment", "mens", "womens", "newbie", "channel")):
    """Descriptive subgroup IPW ATEs with standard errors; no subgroup selection."""
    tau = randomized_pseudo_outcome(df[outcome_col], df[treatment_col], propensity)
    result = {}
    for col in subgroup_cols:
        if col not in df:
            continue
        groups = []
        for value, idx in df.groupby(col, dropna=False).groups.items():
            vals = tau[np.asarray(list(idx), dtype=int)]
            estimate = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
            groups.append({
                "level": str(value), "n": int(len(vals)), "ipw_ate": estimate,
                "ci_lower": estimate - 1.96 * se if se is not None else None,
                "ci_upper": estimate + 1.96 * se if se is not None else None,
            })
        result[col] = groups
    return result


def assignment_probability_stress_test(df, outcome_col, treatment_col="treatment",
                                       known_propensity=2 / 3, deltas=(-0.02, 0.0, 0.02)):
    """Show estimator sensitivity to deliberately perturbed design probabilities.

    This is a stress test around the known design value, not propensity fitting
    and not a claim that the true randomization probability is uncertain.
    """
    y = df[outcome_col].to_numpy(dtype=float)
    t = df[treatment_col].to_numpy(dtype=int)
    rows = []
    for delta in deltas:
        p = float(known_propensity + delta)
        if not 0 < p < 1:
            continue
        estimate = float(np.mean(randomized_pseudo_outcome(y, t, p)))
        rows.append({"assumed_probability": p, "delta_from_known_design": float(delta),
                     "ipw_ate_stress_estimate": estimate,
                     "is_known_design_value": bool(abs(delta) < 1e-12)})
    return {
        "interpretation": "Perturbation stress test only; primary estimate must use the known randomized assignment probability.",
        "results": rows,
    }
