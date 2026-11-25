"""Honest one-period customer-spend prediction with split-conformal intervals.

Hillstrom is a one-period randomized experiment, not a time-indexed panel.
This module therefore performs cross-sectional outcome prediction, not
time-series forecasting. Only pre-treatment customer attributes are features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import FEATURES


def _make_model(seed: int) -> Pipeline:
    numeric = ["recency", "history", "mens", "womens", "newbie"]
    categorical = ["history_segment", "zip_code", "channel"]
    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=250,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2.0,
            random_state=seed,
        )),
    ])


def _metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
    }


def run_spend_prediction(df: pd.DataFrame, seed: int = 42, alpha: float = 0.10,
                         permutation_repeats: int = 5):
    """Fit, calibrate and test a spend prediction model using a 60/20/20 split."""
    required = set(FEATURES) | {"spend", "treatment"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not 0 < alpha < 1 or permutation_repeats < 1:
        raise ValueError("alpha must be in (0,1) and permutation_repeats positive")

    frame = df.copy()
    y = frame["spend"].astype(float).to_numpy()
    treatment = frame["treatment"].astype(int).to_numpy()
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Spend must contain finite, non-negative values")
    if not set(np.unique(treatment)).issubset({0, 1}) or len(np.unique(treatment)) != 2:
        raise ValueError("The pooled binary treatment column must contain both 0 and 1")

    X = frame[FEATURES]
    strata = pd.Series(treatment.astype(str) + "_" + (y > 0).astype(int).astype(str))
    train_cal_idx, test_idx = train_test_split(
        np.arange(len(frame)), test_size=0.20, random_state=seed, stratify=strata
    )
    train_idx, calibration_idx = train_test_split(
        train_cal_idx, test_size=0.25, random_state=seed + 1,
        stratify=strata.iloc[train_cal_idx],
    )
    X_train, X_cal, X_test = X.iloc[train_idx], X.iloc[calibration_idx], X.iloc[test_idx]
    y_train, y_cal, y_test = y[train_idx], y[calibration_idx], y[test_idx]
    t_test = treatment[test_idx]

    model = _make_model(seed)
    model.fit(X_train, y_train)
    calibration_pred = np.maximum(0.0, model.predict(X_cal))
    calibration_residuals = np.abs(y_cal - calibration_pred)
    # Finite-sample split-conformal "higher" quantile.
    rank = min(len(calibration_residuals), int(np.ceil((len(calibration_residuals) + 1) * (1-alpha))))
    radius = float(np.sort(calibration_residuals)[rank - 1])
    prediction = np.maximum(0.0, model.predict(X_test))
    lower = np.maximum(0.0, prediction - radius)
    upper = prediction + radius

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(train_idx), 1)), y_train)
    baseline_pred = np.repeat(float(y_train.mean()), len(test_idx))

    test_metrics = _metrics(y_test, prediction)
    test_metrics["conformal_interval"] = {
        "nominal_coverage": float(1-alpha),
        "empirical_coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
        "mean_width": float(np.mean(upper-lower)),
        "calibration_radius": radius,
        "calibration_n": int(len(calibration_idx)),
        "test_n": int(len(test_idx)),
        "method": "split conformal with absolute residuals and finite-sample higher quantile",
    }

    by_arm = {}
    for arm in (0, 1):
        mask = t_test == arm
        by_arm["email" if arm == 1 else "control"] = {
            **_metrics(y_test[mask], prediction[mask]),
            "conformal_coverage": float(np.mean((y_test[mask] >= lower[mask]) & (y_test[mask] <= upper[mask]))),
            "n": int(mask.sum()),
        }

    importance = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=permutation_repeats, random_state=seed, n_jobs=-1,
    )
    feature_importance = [
        {"feature": feature, "mae_increase": float(value),
         "standard_deviation": float(std)}
        for feature, value, std in sorted(
            zip(FEATURES, importance.importances_mean, importance.importances_std),
            key=lambda item: item[1], reverse=True,
        )
    ]

    results = {
        "study": "Hillstrom cross-sectional spend prediction",
        "task_type": "one-period outcome prediction; not time-series forecasting",
        "dataset_rows": int(len(frame)),
        "features": list(FEATURES),
        "split": {"train": int(len(train_idx)), "calibration": int(len(calibration_idx)),
                  "test": int(len(test_idx)), "seed": int(seed)},
        "model": "HistGradientBoostingRegressor with imputation and one-hot encoding",
        "baseline": {"model": "training-mean DummyRegressor", **_metrics(y_test, baseline_pred)},
        "model_metrics": test_metrics,
        "by_randomized_arm": by_arm,
        "feature_permutation_importance": feature_importance,
        "interpretation": [
            "This is a cross-sectional prediction task because the source has no repeated time index.",
            "Conformal coverage is marginal under exchangeability; it is not a conditional guarantee for every customer segment.",
            "A good spend predictor does not establish heterogeneous treatment-effect or targeting accuracy.",
            "The randomized holdout uplift benchmark remains the source for policy evaluation.",
        ],
    }
    return results, model
