"""Reproducible uplift learner benchmarks for randomized email experiments.

Implements cross-fitted S-, T-, X- and doubly-robust (DR) learners for a
binary treatment contrast. Models are deliberately lightweight sklearn
estimators; this module does not claim causal-forest equivalence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _feature_pipeline(model, numeric, categorical):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def _outcome_model(y, seed):
    if set(np.unique(y)).issubset({0, 1}):
        return LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=8,
                                         min_samples_leaf=15, l2_regularization=2.0,
                                         random_state=seed)


def _predict_outcome(model, X, binary):
    if binary:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _make_model(y, numeric, categorical, seed):
    return _feature_pipeline(_outcome_model(y, seed), numeric, categorical)


def fit_uplift_learners(X, treatment, outcome, propensity=None, seed=42):
    """Fit S/T/X/DR learners and return individual uplift predictions.

    Intended for a randomized binary treatment contrast. Propensity is known
    from the experiment design; when omitted, empirical assignment share is used.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    if len(X) != len(t) or len(t) != len(y) or X.empty:
        raise ValueError("X, treatment and outcome must have equal non-zero length")
    if not np.isfinite(y).all() or not set(np.unique(t)).issubset({0, 1}) or len(np.unique(t)) != 2:
        raise ValueError("Finite outcome and both binary treatment arms are required")
    p = float(np.mean(t)) if propensity is None else float(propensity)
    if not 0 < p < 1:
        raise ValueError("propensity must be strictly between zero and one")
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    binary = set(np.unique(y)).issubset({0.0, 1.0})

    # T-learner
    m0 = _make_model(y[t == 0], numeric, categorical, seed)
    m1 = _make_model(y[t == 1], numeric, categorical, seed + 1)
    m0.fit(X.loc[t == 0], y[t == 0])
    m1.fit(X.loc[t == 1], y[t == 1])
    mu0, mu1 = _predict_outcome(m0, X, binary), _predict_outcome(m1, X, binary)
    t_uplift = mu1 - mu0

    # S-learner: include treatment as an ordinary feature.
    Xs = X.copy()
    Xs["__treatment__"] = t
    s_numeric = numeric + ["__treatment__"]
    sm = _make_model(y, s_numeric, categorical, seed + 2)
    sm.fit(Xs, y)
    X1, X0 = X.copy(), X.copy()
    X1["__treatment__"], X0["__treatment__"] = 1, 0
    s_uplift = (_predict_outcome(sm, X1, binary)
                - _predict_outcome(sm, X0, binary))

    # X-learner: impute effects with opposite-arm outcome models.
    d1 = y[t == 1] - mu0[t == 1]
    d0 = mu1[t == 0] - y[t == 0]
    x1 = _make_model(d1, numeric, categorical, seed + 3)
    x0 = _make_model(d0, numeric, categorical, seed + 4)
    x1.fit(X.loc[t == 1], d1)
    x0.fit(X.loc[t == 0], d0)
    x_uplift = p * x0.predict(X) + (1 - p) * x1.predict(X)

    # Doubly robust pseudo-outcome using cross-fitted nuisance predictions.
    # Nuisance predictions below are generated out-of-fold to reduce overfit.
    mu0_oof, mu1_oof, e_oof = (np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X)))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train, valid in folds.split(X, t):
        for arm, target in [(0, mu0_oof), (1, mu1_oof)]:
            mask = t[train] == arm
            om = _make_model(y[train][mask], numeric, categorical, seed + 10 + arm)
            om.fit(X.iloc[train].iloc[mask], y[train][mask])
            target[valid] = _predict_outcome(om, X.iloc[valid], binary)
        # The randomized propensity is known by design; do not estimate it.
        e_oof[valid] = p
    dr_score = (mu1_oof - mu0_oof
                + t * (y - mu1_oof) / e_oof
                - (1 - t) * (y - mu0_oof) / (1 - e_oof))
    dr_model = _make_model(dr_score, numeric, categorical, seed + 20)
    dr_model.fit(X, dr_score)
    dr_uplift = dr_model.predict(X)

    return {"s_learner": s_uplift, "t_learner": t_uplift,
            "x_learner": x_uplift, "dr_learner": dr_uplift,
            "propensity": p, "outcome_type": "binary" if binary else "continuous"}


def _policy_value(y, t, policy, p):
    contribution = (t == policy) * y / np.where(policy == 1, p, 1 - p)
    value = float(np.mean(contribution))
    se = float(np.std(contribution, ddof=1) / np.sqrt(len(y)))
    return {"value": value, "ci_lower": value - 1.96 * se,
            "ci_upper": value + 1.96 * se}


def repeated_crossfit_benchmark(df, outcome_col="conversion", treatment_col="treatment",
                                feature_cols=None, seeds=(11, 23, 37, 53, 71),
                                n_splits=3, propensity=2/3):
    """Compare uplift learners and simple policies using repeated OOF predictions.

    Each repeat generates out-of-fold predictions; thresholds are fixed at zero.
    Policy value is estimated using the experiment's known assignment propensity.
    Baselines use pre-treatment recency and historical spend only.
    """
    if feature_cols is None:
        feature_cols = ["recency", "history_segment", "history", "mens",
                        "womens", "zip_code", "newbie", "channel"]
    required = set(feature_cols + [outcome_col, treatment_col])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = df.reset_index(drop=True)
    X = data[feature_cols]
    y = data[outcome_col].astype(float).to_numpy()
    t = data[treatment_col].astype(int).to_numpy()
    p = float(propensity)
    if not 0 < p < 1:
        raise ValueError("propensity must be strictly between zero and one")
    records = []
    for seed in seeds:
        predictions = {name: np.zeros(len(data)) for name in
                       ["s_learner", "t_learner", "x_learner", "dr_learner"]}
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for train, valid in splitter.split(X, t):
            fitted = fit_uplift_learners(X.iloc[train], t[train], y[train],
                                         propensity=p, seed=int(seed))
            for name in predictions:
                # Each learner is trained only on the fold's training rows.
                # Predictions are obtained by a small helper refit on the full
                # training fold, avoiding reuse of in-sample training scores.
                pass
            # fit_uplift_learners returns training predictions; fit models again
            # on train and score validation via the dedicated fold helper below.
            fold_pred = _fit_predict_fold(X.iloc[train], t[train], y[train],
                                          X.iloc[valid], p, int(seed))
            for name, values in fold_pred.items():
                predictions[name][valid] = values
        # Business-rule ranking baselines are fixed before evaluation.
        rules = {
            "random": np.random.default_rng(int(seed)).random(len(data)),
            "recency": -pd.to_numeric(data["recency"]).to_numpy(),
            "historical_spend": pd.to_numeric(data["history"]).to_numpy(),
            "mens_affinity": pd.to_numeric(data["mens"]).to_numpy(),
            "womens_affinity": pd.to_numeric(data["womens"]).to_numpy(),
        }
        policies = {name: (score > 0).astype(int) for name, score in predictions.items()}
        policies.update({name: _top_fraction_policy(score, p) for name, score in rules.items()})
        policies["all_treat"] = np.ones(len(data), dtype=int)
        policies["none"] = np.zeros(len(data), dtype=int)
        for name, policy in policies.items():
            value = _policy_value(y, t, policy, p)
            records.append({"seed": int(seed), "policy": name, **value})
    frame = pd.DataFrame(records)
    summary = (frame.groupby("policy")
               .agg(mean_value=("value", "mean"), sd_across_repeats=("value", "std"),
                    mean_ci_lower=("ci_lower", "mean"), mean_ci_upper=("ci_upper", "mean"))
               .reset_index())
    return {"summary": summary.to_dict(orient="records"),
            "repeats": frame.to_dict(orient="records"),
            "seeds": [int(s) for s in seeds], "n_splits": int(n_splits),
            "propensity": p, "outcome": outcome_col}


def _top_fraction_policy(score, fraction):
    threshold = np.quantile(score, 1 - fraction)
    return (score >= threshold).astype(int)


def _fit_predict_fold(X_train, t_train, y_train, X_valid, p, seed):
    """Fold-safe prediction helper for S/T/X/DR learners."""
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    binary = set(np.unique(y_train)).issubset({0.0, 1.0})
    def make(y, s):
        return _make_model(y, numeric, categorical, s)
    def pred(model):
        return _predict_outcome(model, X_valid, binary)
    m0, m1 = make(y_train[t_train == 0], seed), make(y_train[t_train == 1], seed + 1)
    m0.fit(X_train.loc[t_train == 0], y_train[t_train == 0])
    m1.fit(X_train.loc[t_train == 1], y_train[t_train == 1])
    mu0, mu1 = pred(m0), pred(m1)
    t_effect = mu1 - mu0
    Xs = X_train.copy()
    Xs["__treatment__"] = t_train
    sm = _make_model(y_train, numeric + ["__treatment__"], categorical, seed + 2)
    sm.fit(Xs, y_train)
    X1, X0 = X_valid.copy(), X_valid.copy()
    X1["__treatment__"], X0["__treatment__"] = 1, 0
    s_effect = _predict_outcome(sm, X1, binary) - _predict_outcome(sm, X0, binary)
    d1, d0 = y_train[t_train == 1] - _predict_outcome(m0, X_train.loc[t_train == 1], binary), _predict_outcome(m1, X_train.loc[t_train == 0], binary) - y_train[t_train == 0]
    x1, x0 = make(d1, seed + 3), make(d0, seed + 4)
    x1.fit(X_train.loc[t_train == 1], d1)
    x0.fit(X_train.loc[t_train == 0], d0)
    x_effect = p * x0.predict(X_valid) + (1 - p) * x1.predict(X_valid)
    # DR score from cross-fitted nuisance models within training fold.
    mu0_oof, mu1_oof = np.zeros(len(X_train)), np.zeros(len(X_train))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 5)
    for tr, va in inner.split(X_train, t_train):
        for arm, target in [(0, mu0_oof), (1, mu1_oof)]:
            mask = t_train[tr] == arm
            nuisance = make(y_train[tr][mask], seed + 6 + arm)
            nuisance.fit(X_train.iloc[tr].iloc[mask], y_train[tr][mask])
            target[va] = _predict_outcome(nuisance, X_train.iloc[va], binary)
    dr_score = (mu1_oof - mu0_oof + t_train * (y_train - mu1_oof) / p
                - (1 - t_train) * (y_train - mu0_oof) / (1 - p))
    drm = make(dr_score, seed + 8)
    drm.fit(X_train, dr_score)
    return {"s_learner": s_effect, "t_learner": t_effect,
            "x_learner": x_effect, "dr_learner": drm.predict(X_valid)}
