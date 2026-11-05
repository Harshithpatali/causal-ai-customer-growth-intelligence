
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from .data import FEATURES
from .statistics import mean_difference
from scipy.integrate import trapezoid

def _make_preprocessor():
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    categorical = ["history_segment", "zip_code", "channel"]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])

def _classifier(seed):
    return Pipeline([
        ("prep", _make_preprocessor()),
        ("model", LogisticRegression(C=0.5, max_iter=1200, solver="liblinear",
                                     random_state=seed))
    ])

def _positive_spend_model(seed):
    # Positive-spend conditional model. Tree regularization is important because
    # conversion-positive observations are rare in this experiment.
    return Pipeline([
        ("prep", _make_preprocessor()),
        ("model", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.06,
                                                max_leaf_nodes=8, min_samples_leaf=12,
                                                l2_regularization=3.0,
                                                random_state=seed))
    ])

def _expected_spend(classifier, regressor, X):
    probability = np.clip(classifier.predict_proba(X)[:, 1], 0, 1)
    conditional = np.maximum(0, regressor.predict(X))
    return probability * conditional, probability, conditional

def qini_curve(y, treatment, uplift, propensity=2.0/3.0):
    """IPW cumulative uplift-gain diagnostic on a randomized evaluation sample.

    The Horvitz-Thompson pseudo-outcome is A*Y/e - (1-A)*Y/(1-e).
    This is an exploratory ranking diagnostic, not a profit estimate.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    u = np.asarray(uplift, dtype=float)
    p = float(propensity)
    if not 0 < p < 1:
        raise ValueError("Treatment propensity must be between 0 and 1")
    if y.ndim != 1 or t.ndim != 1 or u.ndim != 1 or not (len(y) == len(t) == len(u)):
        raise ValueError("y, treatment and uplift must be one-dimensional arrays of equal length")
    if len(y) == 0:
        return {"fraction": [0.0, 1.0], "gain": [0.0, 0.0],
                "random_gain": [0.0, 0.0], "qini": 0.0, "auuc": 0.0}
    if not np.isfinite(y).all() or not np.isfinite(u).all():
        raise ValueError("y and uplift must contain only finite values")
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("Treatment must be binary (0/1)")

    order = np.argsort(-u, kind="stable")
    y_sorted, t_sorted = y[order], t[order]
    pseudo = (t_sorted * y_sorted / p) - ((1 - t_sorted) * y_sorted / (1 - p))
    gain = np.cumsum(pseudo)
    fraction = np.arange(1, len(y) + 1, dtype=float) / len(y)
    random_gain = fraction * gain[-1]
    x_full = np.concatenate(([0.0], fraction))
    gain_full = np.concatenate(([0.0], gain))
    random_full = np.concatenate(([0.0], random_gain))
    qini = float(trapezoid(gain_full - random_full, x_full))
    auuc = float(trapezoid(gain_full, x_full))
    step = max(1, len(x_full) // 500)
    keep = np.unique(np.concatenate((np.arange(0, len(x_full), step), [len(x_full) - 1])))
    return {
        "fraction": x_full[keep].astype(float).tolist(),
        "gain": gain_full[keep].astype(float).tolist(),
        "random_gain": random_full[keep].astype(float).tolist(),
        "qini": qini, "auuc": auuc, "total_ipw_gain": float(gain[-1]),
    }

def evaluate_policy_ipw(y, treatment, policy, propensity):
    """Evaluate a pre-specified treatment policy on randomized holdout data using IPW."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    policy = np.asarray(policy, dtype=int)
    p = float(propensity)
    if not 0 < p < 1:
        raise ValueError("Treatment propensity must be between 0 and 1")
    if not (len(y) == len(treatment) == len(policy)):
        raise ValueError("y, treatment and policy must have equal lengths")

    selected = (treatment == policy)
    probabilities = np.where(policy == 1, p, 1-p)
    contributions = selected.astype(float) * y / probabilities
    value = float(np.mean(contributions))
    se = float(np.std(contributions, ddof=1) / np.sqrt(len(contributions)))
    return {
        "value": value,
        "standard_error": se,
        "ci_lower": value - 1.96*se,
        "ci_upper": value + 1.96*se,
        "matched_fraction": float(selected.mean()),
        "n": int(len(y))
    }



def compare_policy_ipw(y, treatment, policy_a, policy_b, propensity):
    """Paired IPW contrast between two fixed policies on the same holdout rows."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(treatment, dtype=int)
    a = np.asarray(policy_a, dtype=int)
    b = np.asarray(policy_b, dtype=int)
    if not (y.ndim == t.ndim == a.ndim == b.ndim == 1 and len(y) == len(t) == len(a) == len(b)):
        raise ValueError("All policy comparison arrays must be one-dimensional and have equal lengths")
    if len(y) < 2:
        raise ValueError("At least two holdout observations are required")
    p = float(propensity)
    if not 0 < p < 1:
        raise ValueError("Treatment propensity must be between 0 and 1")
    if not np.isfinite(y).all():
        raise ValueError("Outcome must contain only finite values")
    if not set(np.unique(t)).issubset({0, 1}) or not set(np.unique(a)).issubset({0, 1}) or not set(np.unique(b)).issubset({0, 1}):
        raise ValueError("Treatment and policies must be binary (0/1)")

    def contribution(policy):
        matched = t == policy
        probability = np.where(policy == 1, p, 1 - p)
        return matched.astype(float) * y / probability

    delta = contribution(a) - contribution(b)
    estimate = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {
        "difference": estimate, "standard_error": se,
        "ci_lower": estimate - 1.96 * se, "ci_upper": estimate + 1.96 * se,
        "n": int(len(delta)),
    }

def train_uplift(df, artifact_dir="artifacts", seed=42):
    """Two-part T-learner: conversion probability × conditional positive spend."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    X = data[FEATURES]
    y = data["spend"].astype(float).to_numpy()
    conversion = (y > 0).astype(int)
    t = data["treatment"].astype(int).to_numpy()
    idx_train, idx_test = train_test_split(np.arange(len(data)), test_size=0.25,
                                            random_state=seed, stratify=t)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    c_train = conversion[idx_train]
    t_train, t_test = t[idx_train], t[idx_test]

    control_classifier, treated_classifier = _classifier(seed), _classifier(seed+1)
    control_spend, treated_spend = _positive_spend_model(seed+2), _positive_spend_model(seed+3)

    control_mask, treated_mask = t_train == 0, t_train == 1
    control_classifier.fit(X_train.iloc[control_mask], c_train[control_mask])
    treated_classifier.fit(X_train.iloc[treated_mask], c_train[treated_mask])

    control_positive = control_mask & (y_train > 0)
    treated_positive = treated_mask & (y_train > 0)
    control_spend.fit(X_train.iloc[control_positive], y_train[control_positive])
    treated_spend.fit(X_train.iloc[treated_positive], y_train[treated_positive])

    mu0, p0, conditional0 = _expected_spend(control_classifier, control_spend, X_test)
    mu1, p1, conditional1 = _expected_spend(treated_classifier, treated_spend, X_test)
    uplift = mu1-mu0
    # The experiment randomized customers approximately 2:1 into pooled email vs no email
    # because two email creatives each had a 1/3 allocation.
    treatment_propensity = 2.0/3.0
    curve = qini_curve(y_test, t_test, uplift, propensity=treatment_propensity)
    ate = mean_difference(y_test, t_test)
    policy = (uplift > 0.05).astype(int)
    policy_value = evaluate_policy_ipw(y_test, t_test, policy, treatment_propensity)
    all_email_value = evaluate_policy_ipw(y_test, t_test, np.ones(len(t_test), dtype=int),
                                          treatment_propensity)
    all_control_value = evaluate_policy_ipw(y_test, t_test, np.zeros(len(t_test), dtype=int),
                                            treatment_propensity)
    policy_vs_all_email = compare_policy_ipw(
        y_test, t_test, policy, np.ones(len(t_test), dtype=int), treatment_propensity
    )

    predictions = data.iloc[idx_test][FEATURES].copy()
    predictions["observed_treatment"] = t_test
    predictions["observed_conversion"] = conversion[idx_test]
    predictions["observed_spend"] = y_test
    predictions["predicted_conversion_control"] = p0
    predictions["predicted_conversion_treatment"] = p1
    predictions["predicted_conditional_spend_control"] = conditional0
    predictions["predicted_conditional_spend_treatment"] = conditional1
    predictions["predicted_spend_control"] = mu0
    predictions["predicted_spend_treatment"] = mu1
    predictions["estimated_incremental_spend"] = uplift
    predictions.to_csv(artifact_dir/"test_predictions.csv", index=False)

    joblib.dump({
        "control_classifier": control_classifier,
        "treated_classifier": treated_classifier,
        "control_spend_model": control_spend,
        "treated_spend_model": treated_spend,
        "features": FEATURES, "seed": seed, "model_type": "two_part_t_learner"
    }, artifact_dir/"uplift_model.joblib")

    control_test, treated_test = t_test == 0, t_test == 1
    def safe_auc(y_true, y_score):
        return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None

    def safe_ap(y_true, y_score):
        return float(average_precision_score(y_true, y_score)) if np.sum(y_true) > 0 else None

    metrics = {
        "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
        "test_spend_rmse_control": float(np.sqrt(mean_squared_error(y_test[control_test], mu0[control_test]))),
        "test_spend_rmse_treated": float(np.sqrt(mean_squared_error(y_test[treated_test], mu1[treated_test]))),
        "test_spend_r2_control": float(r2_score(y_test[control_test], mu0[control_test])),
        "test_spend_r2_treated": float(r2_score(y_test[treated_test], mu1[treated_test])),
        "test_conversion_auc_control": safe_auc(conversion[idx_test][control_test], p0[control_test]),
        "test_conversion_auc_treated": safe_auc(conversion[idx_test][treated_test], p1[treated_test]),
        "test_conversion_ap_control": safe_ap(conversion[idx_test][control_test], p0[control_test]),
        "test_conversion_ap_treated": safe_ap(conversion[idx_test][treated_test], p1[treated_test]),
        "test_ate_spend": ate,
        "policy_threshold_expected_incremental_spend": 0.05,
        "policy_value_ipw": policy_value,
        "all_email_value_ipw": all_email_value,
        "all_control_value_ipw": all_control_value,
        "policy_vs_all_email_ipw": policy_vs_all_email,
        "policy_incremental_value_vs_all_control": float(policy_value["value"]-all_control_value["value"]),
        "qini": curve["qini"], "auuc": curve["auuc"],
        "qini_total_ipw_gain": curve.get("total_ipw_gain"),
        "qini_random_gain": curve["random_gain"],
        "qini_fraction": curve["fraction"], "qini_gain": curve["gain"],
        "mean_predicted_uplift": float(np.mean(uplift)),
        "positive_uplift_share": float(np.mean(uplift > 0)),
        "outcome": "spend", "treatment": "email vs no email",
        "model_type": "two_part_t_learner",
        "note": "Expected spend = P(conversion) × E(spend | conversion). Policy values use a randomized holdout; model policy has not demonstrated superiority to email-everyone. Qini is an exploratory IPW cumulative-gain diagnostic."
    }
    (artifact_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def load_models(artifact_dir="artifacts"):
    p = Path(artifact_dir)/"uplift_model.joblib"
    if not p.exists():
        raise FileNotFoundError("Model artifacts not found. Run: python scripts/run_pipeline.py")
    return joblib.load(p)

def score_customers(df, models):
    X = df[FEATURES]
    mu0, p0, conditional0 = _expected_spend(models["control_classifier"],
                                             models["control_spend_model"], X)
    mu1, p1, conditional1 = _expected_spend(models["treated_classifier"],
                                             models["treated_spend_model"], X)
    out = df.copy()
    out["predicted_conversion_control"] = p0
    out["predicted_conversion_treatment"] = p1
    out["predicted_spend_control"] = mu0
    out["predicted_spend_treatment"] = mu1
    out["estimated_incremental_spend"] = mu1-mu0
    return out
