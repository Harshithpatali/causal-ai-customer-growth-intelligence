"""Cross-fitted nuisance-model predictions for doubly robust estimation."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import FEATURES
from .forecasting import _make_model


def cross_fitted_spend_predictions(df, seed=42, folds=5):
    """Produce out-of-fold spend predictions under both randomized actions."""
    X = df[FEATURES]
    y = df["spend"].astype(float).to_numpy()
    treatment = df["treatment"].astype(int).to_numpy()
    mu1, mu0 = np.empty(len(df)), np.empty(len(df))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(X, treatment):
        for arm, target in ((0, mu0), (1, mu1)):
            arm_train = train_idx[treatment[train_idx] == arm]
            model = _make_model(seed + arm)
            model.fit(X.iloc[arm_train], y[arm_train])
            target[valid_idx] = np.maximum(0.0, model.predict(X.iloc[valid_idx]))
    return mu1, mu0
