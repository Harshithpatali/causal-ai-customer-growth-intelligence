
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, average_precision_score
from .data import FEATURES
from .statistics import mean_difference

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

def qini_curve(y, treatment, uplift):
    """Cumulative treated-vs-control outcome-rate ranking diagnostic."""
    frame = pd.DataFrame({"y": np.asarray(y), "t": np.asarray(treatment),
                          "u": np.asarray(uplift)}).sort_values("u", ascending=False)
    nt, nc = (frame.t == 1).sum(), (frame.t == 0).sum()
    if nt == 0 or nc == 0:
        return {"fraction": [0,1], "gain": [0,0], "qini": 0.0, "auuc": 0.0}
    cum_t = (frame.y * (frame.t == 1)).cumsum()
    cum_c = (frame.y * (frame.t == 0)).cumsum()
    seen_t = (frame.t == 1).cumsum()
    seen_c = (frame.t == 0).cumsum()
    rate_t = np.divide(cum_t, seen_t, out=np.zeros(len(frame)), where=seen_t.to_numpy()>0)
    rate_c = np.divide(cum_c, seen_c, out=np.zeros(len(frame)), where=seen_c.to_numpy()>0)
    gain = (rate_t-rate_c) * np.arange(1, len(frame)+1)
    frac = np.arange(1, len(frame)+1)/len(frame)
    gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = np.linspace(0, float(gain[-1]), len(gain))
    qini = float(np.trapezoid(gain-baseline, frac))
    auuc = float(np.trapezoid(gain, frac))
    return {"fraction": [0.0]+frac.tolist(), "gain": [0.0]+gain.tolist(),
            "qini": qini, "auuc": auuc}


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
    curve = qini_curve(y_test, t_test, uplift)
    ate = mean_difference(y_test, t_test)
    # The experiment randomized customers approximately 2:1 into pooled email vs no email
    # because two email creatives each had a 1/3 allocation.
    treatment_propensity = 2.0/3.0
    policy = (uplift > 0.05).astype(int)
    policy_value = evaluate_policy_ipw(y_test, t_test, policy, treatment_propensity)
    all_email_value = evaluate_policy_ipw(y_test, t_test, np.ones(len(t_test), dtype=int),
                                          treatment_propensity)
    all_control_value = evaluate_policy_ipw(y_test, t_test, np.zeros(len(t_test), dtype=int),
                                            treatment_propensity)

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

    def arm_metrics(mask, pred, classifier):
        actual = y_test[mask]
        return {
            "spend_rmse": float(np.sqrt(mean_squared_error(actual, pred[mask]))),
            "spend_r2": float(r2_score(actual, pred[mask])),
            "conversion_roc_auc": float(roc_auc_score(conversion[idx_test][mask],
                                                       classifier.predict_proba(X_test.iloc[np.where(mask)[0]])[:,1]))
                if len(np.unique(conversion[idx_test][mask])) > 1 else None
        }
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
        "policy_incremental_value_vs_all_control": float(policy_value["value"]-all_control_value["value"]),
        "qini": curve["qini"], "auuc": curve["auuc"],
        "qini_fraction": curve["fraction"], "qini_gain": curve["gain"],
        "mean_predicted_uplift": float(np.mean(uplift)),
        "positive_uplift_share": float(np.mean(uplift > 0)),
        "outcome": "spend", "treatment": "email vs no email",
        "model_type": "two_part_t_learner",
        "note": "Expected spend = P(conversion) × E(spend | conversion). Holdout ranking metrics are diagnostics, not proof of policy value."
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
