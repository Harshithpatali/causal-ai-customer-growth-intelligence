"""Repeated-split robustness checks for uplift ranking and policy evaluation.

The routine is deliberately separate from the single frozen train/holdout report:
it runs multiple reproducible stratified splits, re-fits the T-learner on each
training partition, and summarizes variability in held-out policy contrasts.
It is a diagnostic across random partitions of the same dataset, not external
validation and not a substitute for a prospective randomized policy trial.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import FEATURES
from .modeling import (
    _classifier,
    _expected_spend,
    _positive_spend_model,
    compare_policy_ipw,
    evaluate_policy_ipw,
    qini_curve,
)


def _fit_two_part_t_learner(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    """Fit separate conversion and positive-spend models by assigned arm."""
    x_train = train[FEATURES]
    x_test = test[FEATURES]
    y_train = train["spend"].astype(float).to_numpy()
    treatment_train = train["treatment"].astype(int).to_numpy()

    control = treatment_train == 0
    treated = treatment_train == 1

    control_classifier = _classifier(seed)
    treated_classifier = _classifier(seed + 1)
    control_spend = _positive_spend_model(seed + 2)
    treated_spend = _positive_spend_model(seed + 3)

    control_classifier.fit(x_train.loc[control], (y_train[control] > 0).astype(int))
    treated_classifier.fit(x_train.loc[treated], (y_train[treated] > 0).astype(int))

    control_positive = control & (y_train > 0)
    treated_positive = treated & (y_train > 0)
    if control_positive.sum() < 2 or treated_positive.sum() < 2:
        raise ValueError("Each treatment arm needs at least two positive-spend training rows")

    control_spend.fit(x_train.loc[control_positive], y_train[control_positive])
    treated_spend.fit(x_train.loc[treated_positive], y_train[treated_positive])

    mu0, _, _ = _expected_spend(control_classifier, control_spend, x_test)
    mu1, _, _ = _expected_spend(treated_classifier, treated_spend, x_test)
    return mu1 - mu0


def repeated_split_robustness(
    df: pd.DataFrame,
    seeds=(11, 23, 37, 51, 73),
    test_size=0.25,
    propensity=2.0 / 3.0,
    policy_threshold=0.05,
    artifact_path=None,
):
    """Evaluate the same fixed uplift threshold over repeated stratified splits.

    Each split is used only for diagnostic repetition. The held-out partition is
    not used to choose the threshold or tune model hyperparameters.
    """
    required = set(FEATURES) | {"spend", "treatment"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) < 20:
        raise ValueError("At least 20 rows are required for repeated-split diagnostics")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < propensity < 1:
        raise ValueError("propensity must be between 0 and 1")
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < 2 or len(set(seed_list)) != len(seed_list):
        raise ValueError("Provide at least two distinct random seeds")

    rows = []
    data = df.reset_index(drop=True).copy()
    for seed in seed_list:
        train_idx, test_idx = train_test_split(
            np.arange(len(data)),
            test_size=test_size,
            random_state=seed,
            stratify=data["treatment"].astype(int),
        )
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        uplift = _fit_two_part_t_learner(train, test, seed)
        y = test["spend"].astype(float).to_numpy()
        treatment = test["treatment"].astype(int).to_numpy()
        policy = (uplift > policy_threshold).astype(int)

        model_value = evaluate_policy_ipw(y, treatment, policy, propensity)
        all_email = evaluate_policy_ipw(
            y, treatment, np.ones(len(test), dtype=int), propensity
        )
        all_control = evaluate_policy_ipw(
            y, treatment, np.zeros(len(test), dtype=int), propensity
        )
        contrast = compare_policy_ipw(
            y, treatment, policy, np.ones(len(test), dtype=int), propensity
        )
        curve = qini_curve(y, treatment, uplift, propensity=propensity)
        rows.append({
            "seed": seed,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "policy_email_share": float(policy.mean()),
            "mean_predicted_uplift": float(np.mean(uplift)),
            "positive_uplift_share": float(np.mean(uplift > 0)),
            "qini_diagnostic": float(curve["qini"]),
            "auuc_diagnostic": float(curve["auuc"]),
            "model_policy_value": float(model_value["value"]),
            "email_everyone_value": float(all_email["value"]),
            "email_nobody_value": float(all_control["value"]),
            "model_minus_email_everyone": float(contrast["difference"]),
            "contrast_ci_lower": float(contrast["ci_lower"]),
            "contrast_ci_upper": float(contrast["ci_upper"]),
        })

    results = pd.DataFrame(rows)
    summary = {
        "n_splits": int(len(results)),
        "seeds": seed_list,
        "test_size": float(test_size),
        "propensity_email": float(propensity),
        "policy_threshold_expected_incremental_spend": float(policy_threshold),
        "interpretation": (
            "Repeated partitions of one historical randomized dataset. "
            "Split-to-split variability is a robustness diagnostic, not "
            "independent replication, external validation, or proof of policy superiority."
        ),
        "metrics": {},
    }
    numeric = [
        "policy_email_share",
        "mean_predicted_uplift",
        "positive_uplift_share",
        "qini_diagnostic",
        "auuc_diagnostic",
        "model_policy_value",
        "email_everyone_value",
        "email_nobody_value",
        "model_minus_email_everyone",
    ]
    for column in numeric:
        values = results[column].astype(float)
        summary["metrics"][column] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    summary["metrics"]["share_contrasts_ci_cover_zero"] = float(
        ((results["contrast_ci_lower"] <= 0) & (results["contrast_ci_upper"] >= 0)).mean()
    )
    payload = {"summary": summary, "splits": rows}
    if artifact_path is not None:
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2))
    return payload
