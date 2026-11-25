import numpy as np
import pandas as pd

from causal_ai.data import FEATURES
from causal_ai.forecasting import run_spend_prediction


def _sample(seed=8, n=900):
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    history = rng.gamma(2.0, 100.0, n)
    conversion = rng.binomial(1, 0.08 + 0.025 * treatment, n)
    spend = conversion * np.maximum(1, 0.02 * history + treatment * 2 + rng.normal(0, 4, n))
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["A", "B", "C"], n),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural", "Suburban"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multi"], n),
        "treatment": treatment,
        "spend": spend,
    })


def test_spend_prediction_uses_disjoint_splits_and_reports_conformal_coverage():
    report, model = run_spend_prediction(_sample(), seed=9, permutation_repeats=2)
    split = report["split"]
    assert split["train"] + split["calibration"] + split["test"] == report["dataset_rows"]
    assert split["train"] > split["calibration"] > 0
    assert 0 <= report["model_metrics"]["conformal_interval"]["empirical_coverage"] <= 1
    assert report["model_metrics"]["conformal_interval"]["mean_width"] >= 0
    assert len(report["feature_permutation_importance"]) == len(FEATURES)


def test_spend_prediction_rejects_negative_target():
    frame = _sample()
    frame.loc[0, "spend"] = -1
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        run_spend_prediction(frame, permutation_repeats=1)
