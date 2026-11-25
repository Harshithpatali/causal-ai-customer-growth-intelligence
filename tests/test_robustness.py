import numpy as np
import pandas as pd
import pytest

from causal_ai.data import FEATURES
from causal_ai.robustness import repeated_split_robustness


def _small_hillstrom_like_frame(n=240, seed=19):
    rng = np.random.default_rng(seed)
    treatment = np.tile([0, 1], n // 2)
    history = rng.uniform(10, 600, n)
    recency = rng.integers(1, 13, n)
    conversion_probability = np.clip(
        0.035 + 0.00002 * history + treatment * 0.012, 0.01, 0.2
    )
    conversion = rng.binomial(1, conversion_probability)
    spend = np.where(
        conversion == 1,
        rng.gamma(shape=2.0, scale=10.0, size=n),
        0.0,
    )
    return pd.DataFrame({
        "recency": recency,
        "history_segment": rng.choice(["low", "medium", "high"], n),
        "history": history,
        "mens": rng.integers(0, 2, n),
        "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Rural", "Suburban"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "treatment": treatment,
        "spend": spend,
        "conversion": conversion,
        "visit": conversion,
    })[FEATURES + ["treatment", "spend", "conversion", "visit"]]


def test_repeated_split_robustness_is_reproducible_and_writes_report(tmp_path):
    frame = _small_hillstrom_like_frame()
    output = tmp_path / "reports" / "repeated_split.json"
    first = repeated_split_robustness(
        frame, seeds=(11, 23), artifact_path=output
    )
    second = repeated_split_robustness(frame, seeds=(11, 23))

    assert first == second
    assert output.exists()
    assert first["summary"]["n_splits"] == 2
    assert len(first["splits"]) == 2
    assert "standard_deviation" in first["summary"]["metrics"]["model_minus_email_everyone"]


def test_repeated_split_robustness_requires_distinct_seeds():
    frame = _small_hillstrom_like_frame()
    with pytest.raises(ValueError, match="distinct"):
        repeated_split_robustness(frame, seeds=(7, 7))


def test_repeated_split_robustness_rejects_missing_model_features():
    frame = _small_hillstrom_like_frame().drop(columns=["history"])
    with pytest.raises(ValueError, match="Missing required columns"):
        repeated_split_robustness(frame, seeds=(11, 23))
