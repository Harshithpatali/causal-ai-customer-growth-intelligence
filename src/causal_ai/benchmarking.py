"""Meta-learner and targeting benchmark on the randomized Hillstrom experiment.

All learners are evaluated on the same stratified holdout. Policy comparisons use
the known pooled assignment probability (2/3). Qini/AUUC use the Horvitz-
Thompson pseudo-outcome and a straight-line random-ranking reference. The
multi-treatment analysis preserves the original three randomized arms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES
from .modeling import compare_policy_ipw, evaluate_policy_ipw, qini_curve

SEGMENTS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")


def _regressor(seed: int, include_treatment=False):
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    if include_treatment:
        numeric.append("__treatment")
    categorical = ["history_segment", "zip_code", "channel"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("model", ExtraTreesRegressor(
            n_estimators=160, min_samples_leaf=8, max_features=0.9,
            random_state=seed, n_jobs=-1,
        )),
    ])


def _check_frame(df):
    required = set(FEATURES) | {"segment", "spend", "conversion", "visit"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not set(df["segment"].dropna().unique()).issubset(set(SEGMENTS)):
        raise ValueError("segment must contain only No E-Mail, Mens E-Mail, and Womens E-Mail")
    if df["segment"].isna().any() or df["spend"].isna().any():
        raise ValueError("segment and spend cannot contain missing values")
    if len(df) < 60:
        raise ValueError("At least 60 rows are required for train/holdout benchmarking")


def _fit_binary_learners(train: pd.DataFrame, seed: int):
    x = train[FEATURES].copy()
    y = train["spend"].astype(float).to_numpy()
    a = (train["segment"] != "No E-Mail").astype(int).to_numpy()
    m0 = _regressor(seed)
    m1 = _regressor(seed + 1)
    m0.fit(x.loc[a == 0], y[a == 0])
    m1.fit(x.loc[a == 1], y[a == 1])

    # S-learner: one outcome model with randomized treatment as an input.
    xs = x.copy()
    xs["__treatment"] = a
    s_model = _regressor(seed + 2, include_treatment=True)
    s_model.fit(xs, y)
    x0, x1 = x.copy(), x.copy()
    x0["__treatment"] = 0
    x1["__treatment"] = 1
    s_uplift = s_model.predict(x1) - s_model.predict(x0)

    # X-learner: impute treatment effects in each arm, fit effect models,
    # then combine with the known treatment propensity.
    tau_treated = y[a == 1] - m0.predict(x.loc[a == 1])
    tau_control = m1.predict(x.loc[a == 0]) - y[a == 0]
    tau1_model = _regressor(seed + 3)
    tau0_model = _regressor(seed + 4)
    tau1_model.fit(x.loc[a == 1], tau_treated)
    tau0_model.fit(x.loc[a == 0], tau_control)
    e = 2.0 / 3.0
    x_uplift = e * tau0_model.predict(x) + (1.0 - e) * tau1_model.predict(x)

    return {
        "T-learner": m1.predict(x) - m0.predict(x),
        "S-learner": s_uplift,
        "X-learner": x_uplift,
    }


def _bootstrap_difference(y, treatment, policy_a, policy_b, propensity,
                          n_bootstrap=1000, seed=2026):
    """Paired nonparametric row bootstrap of the IPW policy-value contrast."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    policy_a = np.asarray(policy_a, dtype=int)
    policy_b = np.asarray(policy_b, dtype=int)
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    point = compare_policy_ipw(y, treatment, policy_a, policy_b, propensity)
    rng = np.random.default_rng(seed)
    n = len(y)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            estimates.append(compare_policy_ipw(
                y[idx], treatment[idx], policy_a[idx], policy_b[idx], propensity
            )["difference"])
        except ValueError:
            continue
    if len(estimates) < 0.9 * n_bootstrap:
        raise RuntimeError("Too many invalid bootstrap replicates")
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimate": float(point["difference"]),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "bootstrap_replicates": int(len(estimates)),
        "bootstrap_standard_error": float(np.std(estimates, ddof=1)),
        "method": "paired nonparametric bootstrap of holdout rows; percentile 95% interval",
    }


def _top_fraction_policy(score, fraction):
    score = np.asarray(score, dtype=float)
    n_select = int(np.ceil(len(score) * fraction))
    policy = np.zeros(len(score), dtype=int)
    if n_select:
        chosen = np.argsort(-score, kind="stable")[:n_select]
        policy[chosen] = 1
    return policy


def _creative_effect_models(train, test, seed):
    """Three-arm T-learner for expected spend under each randomized creative."""
    models = {}
    for offset, segment in enumerate(SEGMENTS):
        subset = train.loc[train["segment"] == segment]
        if len(subset) < 10:
            raise ValueError(f"Insufficient training observations in arm: {segment}")
        model = _regressor(seed + offset)
        model.fit(subset[FEATURES], subset["spend"].astype(float))
        models[segment] = model
    predictions = {segment: model.predict(test[FEATURES]) for segment, model in models.items()}
    return models, predictions


def _welch_difference_ci(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    difference = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / len(a) + vb / len(b)
    se = float(np.sqrt(se2))
    df = se2**2 / ((va / len(a))**2 / (len(a) - 1) +
                   (vb / len(b))**2 / (len(b) - 1))
    critical = float(student_t.ppf(0.975, df))
    return {
        "difference": difference,
        "standard_error": se,
        "ci_lower": float(difference - critical * se),
        "ci_upper": float(difference + critical * se),
        "n_a": int(len(a)), "n_b": int(len(b)),
        "estimator": "unadjusted randomized-arm difference in means; Welch 95% CI",
    }


def run_uplift_benchmark(
    df: pd.DataFrame,
    test_size=0.25,
    seed=42,
    n_bootstrap=1000,
    artifact_path=None,
):
    """Run meta-learner, heuristic-ranking, bootstrap and three-arm comparisons."""
    _check_frame(df)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    data = df.reset_index(drop=True).copy()
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=test_size, random_state=seed,
        stratify=data["segment"],
    )
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    uplift_scores = _fit_binary_learners(train, seed)
    y = test["spend"].astype(float).to_numpy()
    a = (test["segment"] != "No E-Mail").astype(int).to_numpy()
    propensity = 2.0 / 3.0

    # All ranking strategies receive the same contact budget: top 2/3,
    # matching the pooled randomized email allocation.
    rankings = dict(uplift_scores)
    rankings["Random (seeded)"] = np.random.default_rng(seed).random(len(test))
    rankings["Recency (more recent first)"] = -pd.to_numeric(test["recency"]).to_numpy()
    rankings["Historical spend (highest first)"] = pd.to_numeric(test["history"]).to_numpy()

    ranking_results = {}
    policies = {}
    for name, score in rankings.items():
        policy = _top_fraction_policy(score, propensity)
        policies[name] = policy
        value = evaluate_policy_ipw(y, a, policy, propensity)
        curve = qini_curve(y, a, score, propensity=propensity)
        ranking_results[name] = {
            "contact_share": float(policy.mean()),
            "policy_value_ipw": value,
            "auuc_ht": curve["auuc"],
            "qini_ht": curve["qini"],
            "qini_definition": (
                "Trapezoidal area between cumulative Horvitz-Thompson pseudo-outcome "
                "gain sorted by score and the straight-line random-ranking gain."
            ),
            "auuc_definition": (
                "Trapezoidal area under cumulative Horvitz-Thompson pseudo-outcome "
                "gain, with x equal to the fraction of holdout customers targeted."
            ),
        }

    bootstrap = {}
    for name, policy in policies.items():
        if name == "Email everyone":
            continue
        bootstrap[name + " vs Email everyone"] = _bootstrap_difference(
            y, a, policy, np.ones(len(test), dtype=int), propensity,
            n_bootstrap=n_bootstrap, seed=seed + 900,
        )

    pairwise_ranking_bootstrap = {}
    ranking_names = list(policies)
    for i, first_name in enumerate(ranking_names):
        for second_name in ranking_names[i + 1:]:
            pairwise_ranking_bootstrap[first_name + " vs " + second_name] = _bootstrap_difference(
                y, a, policies[first_name], policies[second_name], propensity,
                n_bootstrap=n_bootstrap, seed=seed + 1200 + i * 31 + ranking_names.index(second_name),
            )

    # Creative-specific randomized-arm estimates and model-implied effects.
    creative_models, creative_predictions = _creative_effect_models(train, test, seed + 100)
    creative_effects = {}
    pairwise = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]
    for first, second in pairwise:
        first_mask = test["segment"].to_numpy() == first
        second_mask = test["segment"].to_numpy() == second
        observed = _welch_difference_ci(
            y[first_mask], y[second_mask]
        )
        model_effect = creative_predictions[first] - creative_predictions[second]
        creative_effects[first + " vs " + second] = {
            "randomized_arm_spend_effect": observed,
            "mean_model_implied_spend_effect": float(np.mean(model_effect)),
            "model_implied_positive_effect_share": float(np.mean(model_effect > 0)),
            "interpretation": (
                "Observed contrast is a randomized-arm average effect; model-implied "
                "customer-level effects are exploratory and are not individually observed."
            ),
        }

    # Pairwise targeting policies compare each creative against control among
    # the two relevant arms. Within each pair the assignment probability is 1/2.
    creative_policy = {}
    for creative in ("Mens E-Mail", "Womens E-Mail"):
        pair = test[test["segment"].isin([creative, "No E-Mail"])].copy()
        score = (
            creative_predictions[creative][test.index.get_indexer(pair.index)]
            - creative_predictions["No E-Mail"][test.index.get_indexer(pair.index)]
        )
        pair_y = pair["spend"].to_numpy(dtype=float)
        pair_t = (pair["segment"].to_numpy() == creative).astype(int)
        pair_score = _top_fraction_policy(score, 0.5)
        value = evaluate_policy_ipw(pair_y, pair_t, pair_score, 0.5)
        everyone = evaluate_policy_ipw(pair_y, pair_t, np.ones(len(pair), dtype=int), 0.5)
        diff = _bootstrap_difference(
            pair_y, pair_t, pair_score, np.ones(len(pair), dtype=int), 0.5,
            n_bootstrap=n_bootstrap, seed=seed + (111 if creative.startswith("Mens") else 222),
        )
        creative_policy[creative + " vs No E-Mail"] = {
            "targeting_policy_value_ipw": value,
            "email_everyone_value_ipw": everyone,
            "targeted_minus_everyone_bootstrap": diff,
            "contact_share_within_pair": float(pair_score.mean()),
            "propensity_within_pair": 0.5,
        }

    payload = {
        "design": {
            "dataset": "Hillstrom / MineThatData Email Analytics Challenge",
            "n_total": int(len(data)), "n_train": int(len(train)), "n_holdout": int(len(test)),
            "seed": int(seed), "test_size": float(test_size),
            "randomized_arms": list(SEGMENTS),
            "pooled_email_propensity": 2.0 / 3.0,
            "within_creative_pair_propensity": 0.5,
            "policy_budget_fraction": 2.0 / 3.0,
        },
        "estimators": {
            "T-learner": "Separate ExtraTrees expected-spend outcome models in pooled email and control arms; uplift = mu1(x)-mu0(x).",
            "S-learner": "One ExtraTrees expected-spend model with treatment indicator; predict each row under treatment=1 and treatment=0, then subtract.",
            "X-learner": "T-learner outcome models impute treated and control conditional effects; separate ExtraTrees effect models are combined using known pooled propensity e=2/3.",
            "policy_value": "Holdout Horvitz-Thompson IPW value: mean[I(A=policy(X))*Y / P(A=policy(X))].",
            "policy_contrast": "Paired difference of per-row IPW contributions on the same randomized holdout.",
            "bootstrap": "Paired nonparametric bootstrap resampling holdout rows; percentile 95% interval. This quantifies holdout sampling uncertainty conditional on the fitted policies, not model-training uncertainty.",
            "qini": "Area between sorted cumulative HT pseudo-outcome gain and the straight-line random-ranking reference.",
            "auuc": "Area under sorted cumulative HT pseudo-outcome gain. Both are ranking diagnostics, not monetary profit or proof of CATE calibration.",
        },
        "binary_uplift_benchmark": ranking_results,
        "bootstrap_policy_contrasts_vs_email_everyone": bootstrap,
        "pairwise_ranking_policy_bootstrap_contrasts": pairwise_ranking_bootstrap,
        "multi_treatment_average_effects": creative_effects,
        "multi_treatment_targeting": creative_policy,
        "limitations": [
            "All binary meta-learners are trained on pooled email versus control; creative-specific effects are separately modelled in a three-arm analysis.",
            "The holdout is one random partition of the same historical experiment; no external validation is claimed.",
            "Bootstrap intervals condition on fitted models and do not include training-sample or model-selection uncertainty.",
            "Random ranking is one seeded realization; repeat seeds for a distribution of random-policy performance.",
            "Recency uses lower numeric values as more recent; verify source coding before operational interpretation.",
            "Customer-level effects are model-implied and not individually observed.",
            "No contact costs or contribution margins are available; policy value is spend, not profit.",
        ],
    }
    if artifact_path is not None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    return payload
