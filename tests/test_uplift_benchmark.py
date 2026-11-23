import numpy as np
import pandas as pd

from causal_ai.uplift_benchmark import (
    fit_uplift_learners,
    repeated_crossfit_benchmark,
    repeated_multitreatment_benchmark,
)


def synthetic_email_data(n=360, seed=9):
    rng = np.random.default_rng(seed)
    treatment = np.tile([0, 1], n // 2)
    history = rng.normal(100, 25, n)
    mens = rng.integers(0, 2, n)
    womens = rng.integers(0, 2, n)
    logits = -2.0 + 0.012 * (history - 100) + 0.5 * treatment * mens
    probability = 1 / (1 + np.exp(-logits))
    conversion = rng.binomial(1, probability)
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["low", "mid", "high"], n),
        "history": history, "mens": mens, "womens": womens,
        "zip_code": rng.choice(["Urban", "Rural"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone"], n),
        "treatment": treatment, "conversion": conversion,
    })


def test_fit_uplift_learners_returns_finite_scores():
    df = synthetic_email_data()
    X = df[["recency", "history_segment", "history", "mens", "womens",
            "zip_code", "newbie", "channel"]]
    result = fit_uplift_learners(X, df.treatment, df.conversion,
                                 propensity=0.5, seed=13)
    for key in ["s_learner", "t_learner", "x_learner", "dr_learner"]:
        assert result[key].shape == (len(df),)
        assert np.isfinite(result[key]).all()


def test_repeated_crossfit_benchmark_is_reproducible_and_includes_baselines():
    df = synthetic_email_data()
    kwargs = dict(outcome_col="conversion", seeds=(5,), n_splits=3, propensity=0.5)
    first = repeated_crossfit_benchmark(df, **kwargs)
    second = repeated_crossfit_benchmark(df, **kwargs)
    assert first == second
    names = {row["policy"] for row in first["summary"]}
    assert {"s_learner", "t_learner", "x_learner", "dr_learner",
            "random", "recency", "historical_spend", "all_treat", "none"} <= names


def test_multitreatment_benchmark_separates_creatives():
    df = synthetic_email_data()
    # Construct the three original arms while preserving a no-email control.
    df["segment"] = np.where(df.treatment.eq(0), "No E-Mail",
                             np.where(df.mens.eq(1), "Mens E-Mail", "Womens E-Mail"))
    result = repeated_multitreatment_benchmark(df, seeds=(3,), n_splits=3)
    assert set(result) == {"Mens E-Mail", "Womens E-Mail"}
    for contrast in result.values():
        assert contrast["outcome"] == "conversion"
