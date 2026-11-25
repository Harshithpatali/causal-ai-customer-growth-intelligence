import numpy as np
import pandas as pd

from causal_ai.data import FEATURES
from causal_ai.advanced_statistics import cross_fitted_spend_predictions


def test_cross_fitted_spend_predictions_are_finite_and_aligned():
    rng = np.random.default_rng(12)
    n = 240
    treatment = np.tile([0, 1], n // 2)
    history = rng.uniform(20, 400, n)
    conversion = rng.binomial(1, 0.15, n)
    frame = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["A", "B", "C"], n),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural", "Suburban"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multi"], n),
        "treatment": treatment,
        "spend": conversion * np.maximum(1, history * 0.02 + treatment),
    })
    mu1, mu0 = cross_fitted_spend_predictions(frame, seed=4, folds=3)
    assert len(mu1) == len(mu0) == len(frame)
    assert np.isfinite(mu1).all() and np.isfinite(mu0).all()
    assert np.all(mu1 >= 0) and np.all(mu0 >= 0)
